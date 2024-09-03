#!python

import argparse
import contextlib
import functools
import os
import re
import sys
from collections import defaultdict

import attrs
from attrs import define, field, asdict

from mako.lookup import TemplateLookup
from mako.runtime import _render
from mako.template import DefTemplate
from mako import exceptions


def to_python_type(type):
    type_name = type[1:]

    match type_name:
        case "float":
            name = "Float"
        case "double":
            name = "Float"
        case "long":
            name = "Long"
        case _:
            raise Exception(f"Unknown Python scalar type for {type}")
    return name


def to_numpy_type(type):
    type_name = type[1:]

    match type_name:
        case "int":
            name = "NPY_INT"
        case "double":
            name = "NPY_DOUBLE"
        case "float":
            name = "NPY_FLOAT"
        case "long":
            name = "NPY_LONG"
        case "float16":
            name = "NPY_HALF"
        case _:
            raise Exception(f"Unknown numpy type for {type_name}")
    return name


next_slot = 1


@define(kw_only=True)
class BinOp:
    left_type: str
    right_type: str

    # force a promotion of the operand to a specific NumPy type
    # canonicalizing commutative operands happens before a possible promotion
    left_promotion: str = None
    right_promotion: str = None
    result_type: str

    # like the above, but does not force promotion if the Python scalar does not fit into the NumPy type
    # this is useful, e.g., when trying to promote a Python float (which can be both a float or a double) to a
    # Numpy float array
    left_conditional_promotion: bool = False
    right_conditional_promotion: bool = False

    # should the operation overwrite the left operand?
    inplace: bool = False

    # is the operation commutative? If so, try to move scalar operands to the right
    commutative: bool = True

    opname: str = ""
    operation: str
    loop_function: str

    impl_template: str = "arith_binop.mako"
    guard_template: str = "binop_case_guard.mako"
    flatten: bool = False

    # cache various intermediate results per instruction occurence
    locality_cache: bool = True

    # disabled, we adaptively free the cache instead
    # locality_cache_size_limit: int = os.sysconf('SC_PAGE_SIZE')

    # maintain cache statistics and print after this many instruction occurences
    locality_stats: bool = False

    # maintain a cache and statisticst, but never use the cache, only analyze locality
    analyze_locality: bool = False

    def __attrs_post_init__(self):
        if self.inplace and not self.operation.startswith("inplace_"):
            self.operation = f"inplace_{self.operation}"

        if self.opname == "":
            self.opname = f"cmlq_{self.left_type}_{self.operation}_{self.right_type}"


    @property
    def with_broadcast_cache_variant(self):
        return (self.is_python_scalar(self.left_type) or
         self.is_python_scalar(self.right_type))


    def is_python_scalar(self, type):
        return type.startswith("s")

    def to_template_args(self):
        args = attrs.asdict(self)
        if self.is_python_scalar(self.left_type):
            args["left_scalar_name"] = to_python_type(self.left_type)
        else:
            args["left_numpy_name"] = to_numpy_type(self.left_type)

        if self.is_python_scalar(self.right_type):
            args["right_scalar_name"] = to_python_type(self.right_type)
        else:
            args["right_numpy_name"] = to_numpy_type(self.right_type)
        signature = self.signature()
        if self.flatten:
            signature = f"__attribute__((flatten)) {signature}"
        args["signature"] = signature
        args["slot_name"] = self.slot_name()
        args["with_broadcast_cache_variant"] = self.with_broadcast_cache_variant

        return args

    def slot_name(self):
        return f"SLOT_{self.left_type.upper()}_{self.operation.upper()}_{self.right_type.upper()}"

    def signature(self):
        return f"""int {self.opname}(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr)"""

    def slot_define(self):
        global next_slot
        next_slot += 1
        return f"""#define {self.slot_name()} {next_slot - 1}"""

    def build_variants(self):
        if self.with_broadcast_cache_variant:
            return [self, ScalarBroadcastBinop(**attrs.asdict(self))]

        return [self]


@define(kw_only=True)
class ScalarBroadcastBinop(BinOp):

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        self.opname = (
            f"cmlq_{self.left_type}_{self.operation}_{self.right_type}_broadcast_cache"
        )

        # we don't want to generate locality cache code, just the cache element access
        self.locality_cache = False

        # handling the broadcast cache case is done in the general template already
        self.guard_template = None

    def slot_name(self):
        return f"SLOT_{self.left_type.upper()}_{self.operation.upper()}_{self.right_type.upper()}_BROADCAST_CACHE"

    def to_template_args(self):
        args = super().to_template_args()
        args["cache_broadcast_array"] = True
        return args


@define(kw_only=True)
class FunctionBinOp(BinOp):
    def __attrs_post_init__(self):
        if self.opname == "":
            self.opname = f"cmlq_{self.operation}_{self.left_type}_{self.right_type}"


    @property
    def with_broadcast_cache_variant(self):
        return False

    def slot_name(self):
        return f"SLOT_{self.operation.upper()}_{self.left_type.upper()}_{self.right_type.upper()}"


@define(kw_only=True)
class ArrayPowerOp(BinOp):
    fixed_exponent: float = None

    fixed_names = {
        2: "square",
    }

    @property
    def with_broadcast_cache_variant(self):
        return False


    def __attrs_post_init__(self):
        if self.fixed_exponent is not None:
            self.opname = f"cmlq_{self.left_type}_{self.fixed_names[self.fixed_exponent]}_{self.operation}_{self.right_type}"
        super().__attrs_post_init__()

    def build_variants(self):
        return [self]


@contextlib.contextmanager
def smart_open(filename=None):
    if filename and filename != "-":
        fh = open(filename, "w")
    else:
        fh = sys.stdout

    try:
        yield fh
    finally:
        if fh is not sys.stdout:
            fh.close()


declaration = """
#define SLOT_ADOUBLE_${operation}_ADOUBLE NEXT_SLOT
"""


def build_derivatives(flatten, cache_stats):
    derivatives = [
        BinOp(
            operation="subtract",
            left_type="afloat",
            right_type="afloat",
            result_type="NPY_FLOAT",
            loop_function="FLOAT_subtract",
            commutative=False,
        ),
        BinOp(
            operation="subtract",
            left_type="afloat",
            right_type="afloat",
            result_type="NPY_FLOAT",
            loop_function="FLOAT_subtract",
            commutative=False,
            inplace=True,
        ),
        BinOp(
            operation="add",
            left_type="afloat",
            right_type="afloat",
            result_type="NPY_FLOAT",
            loop_function="FLOAT_add",
        ),
        BinOp(
            operation="add",
            left_type="afloat",
            right_type="afloat",
            result_type="NPY_FLOAT",
            loop_function="FLOAT_add",
            inplace=True,
        ),
        BinOp(
            operation="multiply",
            left_type="afloat",
            right_type="afloat",
            result_type="NPY_FLOAT",
            loop_function="FLOAT_multiply",
        ),


        # These cases lead to issues with legacy, value-based promotion. Small long or float values are promoted to
        # float arrays, but our optimization always promotes them to long/double arrays. This leads to incorrect
        # strides. We don't want the whole value-based promotion logic in the derivatives, but we could perform the
        # value-based promotion at rewrite-time for variants with cached broadcast arrays (i.e. where one of the
        # operands is constant). As the cost of value-based promotion is paid only once there, it could be profitable.
        # This is not implemented yet, so we disable these cases for now.

        # Another issue is that even without value-based promotion, the Python scalar is promoted to an array that can
        # fit the value of the scalar. In other words, the exact array type we need depends on a runtime value. These
        # cases could still be optimized by introducing a guard for the value range, but this is not implemented yet.

        # BinOp(
        #     operation="add",
        #     left_type="afloat",
        #     right_type="sfloat",
        #     result_type="NPY_FLOAT",
        #     loop_function="FLOAT_add",
        # ),
        #
        # BinOp(
        #     operation="subtract",
        #     left_type="afloat",
        #     right_type="sfloat",
        #     result_type="NPY_FLOAT",
        #     loop_function="FLOAT_subtract",
        #     commutative=False,
        # ),
        #
        # BinOp(
        #     operation="subtract",
        #     left_type="slong",
        #     right_type="afloat",
        #     result_type="NPY_FLOAT",
        #     loop_function="FLOAT_subtract",
        #     commutative=False,
        # ),
        # BinOp(
        #     operation="multiply",
        #     left_type="afloat",
        #     right_type="slong",
        #     result_type="NPY_FLOAT",
        #     loop_function="FLOAT_multiply",
        # ),
        # BinOp(
        #     operation="multiply",
        #     left_type="afloat",
        #     right_type="sfloat",
        #     result_type="NPY_FLOAT",
        #     loop_function="FLOAT_multiply",
        # ),

        BinOp(
            operation="subtract",
            left_type="adouble",
            right_type="adouble",
            result_type="NPY_DOUBLE",
            loop_function="DOUBLE_subtract",
            commutative=False,
        ),
        BinOp(
            operation="subtract",
            left_type="adouble",
            right_type="adouble",
            result_type="NPY_DOUBLE",
            loop_function="DOUBLE_subtract",
            commutative=False,
            inplace=True,
        ),
        BinOp(
            operation="subtract",
            left_type="adouble",
            right_type="sfloat",
            result_type="NPY_DOUBLE",
            loop_function="DOUBLE_subtract",
            commutative=False,
        ),
        BinOp(
            operation="add",
            left_type="adouble",
            right_type="adouble",
            result_type="NPY_DOUBLE",
            loop_function="DOUBLE_add",
        ),
        BinOp(
            operation="add",
            left_type="adouble",
            right_type="adouble",
            result_type="NPY_DOUBLE",
            loop_function="DOUBLE_add",
            inplace=True,
        ),
        BinOp(
            operation="add",
            left_type="adouble",
            right_type="sfloat",
            result_type="NPY_DOUBLE",
            loop_function="DOUBLE_add",
        ),
        BinOp(
            operation="add",
            left_type="adouble",
            right_type="sfloat",
            result_type="NPY_DOUBLE",
            loop_function="DOUBLE_add",
            inplace=True
        ),
        BinOp(
            operation="multiply",
            left_type="adouble",
            right_type="adouble",
            result_type="NPY_DOUBLE",
            loop_function="DOUBLE_multiply",
        ),
        BinOp(
            operation="multiply",
            left_type="adouble",
            right_type="slong",
            result_type="NPY_DOUBLE",
            loop_function="DOUBLE_multiply",
            # numpy promotion rules promote the long to a double array (instead of a long array)
            right_promotion="NPY_DOUBLE",
        ),
        BinOp(
            operation="multiply",
            left_type="adouble",
            right_type="sfloat",
            result_type="NPY_DOUBLE",
            loop_function="DOUBLE_multiply",
        ),
        BinOp(
            operation="multiply",
            left_type="adouble",
            right_type="sfloat",
            result_type="NPY_DOUBLE",
            loop_function="DOUBLE_multiply",
            inplace=True
        ),
        BinOp(
            operation="multiply",
            left_type="along",
            right_type="slong",
            result_type="NPY_LONG",
            loop_function="LONG_multiply",
        ),

        # this case requires potential casting of the long array, which is not currently implemented
        # see check_for_trivial_loop in multiarraymodule.c
        # BinOp(
        #     operation="multiply",
        #     left_type="along",
        #     right_type="sfloat",
        #     result_type="NPY_DOUBLE",
        #     loop_function="DOUBLE_multiply",
        #     left_promotion="NPY_DOUBLE",
        # ),
        BinOp(
            operation="true_divide",
            left_type="along",
            right_type="sfloat",
            result_type="NPY_DOUBLE",
            loop_function="DOUBLE_divide",
            commutative=False,
        ),
        BinOp(
            operation="true_divide",
            left_type="adouble",
            right_type="sfloat",
            result_type="NPY_DOUBLE",
            loop_function="DOUBLE_divide",
            commutative=False,
        ),
        BinOp(
            operation="true_divide",
            left_type="adouble",
            right_type="slong",
            result_type="NPY_DOUBLE",
            loop_function="DOUBLE_divide",
            right_promotion="NPY_DOUBLE",
            commutative=False,
        ),
        BinOp(
            operation="true_divide",
            left_type="sfloat",
            right_type="adouble",
            result_type="NPY_DOUBLE",
            loop_function="DOUBLE_divide",
            commutative=False,
        ),
        BinOp(
            operation="true_divide",
            left_type="adouble",
            right_type="adouble",
            result_type="NPY_DOUBLE",
            loop_function="DOUBLE_divide",
            commutative=False,
        ),
        ArrayPowerOp(
            operation="power",
            left_type="adouble",
            right_type="sfloat",
            result_type="NPY_DOUBLE",
            loop_function="DOUBLE_power",
            impl_template="array_power.mako",
            commutative=False,
        ),
        ArrayPowerOp(
            operation="power",
            left_type="adouble",
            right_type="sfloat",
            result_type="NPY_DOUBLE",
            loop_function="DOUBLE_square",
            impl_template="array_power.mako",
            guard_template="array_power_case_guard.mako",
            commutative=False,
            fixed_exponent=2.0,
        ),
        ArrayPowerOp(
            operation="power",
            left_type="adouble",
            right_type="slong",
            result_type="NPY_DOUBLE",
            loop_function="DOUBLE_square",
            impl_template="array_power.mako",
            guard_template="array_power_case_guard.mako",
            commutative=False,
            fixed_exponent=2,
        ),
        ArrayPowerOp(
            operation="power",
            left_type="adouble",
            right_type="slong",
            result_type="NPY_DOUBLE",
            loop_function="DOUBLE_power",
            impl_template="array_power.mako",
            commutative=False,
        ),
        FunctionBinOp(
            operation="minimum",
            left_type="aint",
            right_type="aint",
            result_type="NPY_INT",
            loop_function="INT_minimum",
            impl_template="function_binop.mako",
        ),
    ]


    all_derivatives = []
    for derivative in derivatives:
        derivative.flatten = flatten
        derivative.locality_stats = cache_stats
        all_derivatives.extend(derivative.build_variants())

    return all_derivatives


collapse_newlines = re.compile("\\n\\s*\\n")


def render_template(template, template_args, out):
    try:
        print(
            re.sub(collapse_newlines, "\n\n", template.render(**template_args)),
            file=out,
        )
    except:
        # Try to re-create the error using a proper file template
        # This will give a clearer error message.
        with open("failed_template.py", "w") as failed:
            failed.write(template._code)
        import failed_template

        data = dict(callable=failed_template.render_body, **template_args)
        try:
            _render(
                DefTemplate(template, failed_template.render_body),
                failed_template.render_body,
                [],
                data,
            )
        except:
            write_html_error()
            raise


def write_html_error():
    print(exceptions.text_error_template().render())
    with open(
        os.path.join(script_dir, "cmlq_templates/error.html"), "wb"
    ) as error_file:
        error_file.write(exceptions.html_error_template().render())


def generate_implementations(derivatives, lookup, out):
    for derivative in derivatives:
        template = lookup.get_template(derivative.impl_template)
        template_args = derivative.to_template_args()
        render_template(template, template_args, out)


def generate_case_guards(derivatives, lookup, out):
    global print
    print = functools.partial(print, file=out)

    binops = [
        d
        for d in derivatives
        if isinstance(d, BinOp) and not isinstance(d, FunctionBinOp)
    ]
    groups = defaultdict(list)
    for binop in binops:
        name = binop.operation
        groups[name].append(binop)

    for group_name, group in groups.items():
        case_name = f"NB_{group_name.upper()}"
        print(
            f"case {case_name}:",
        )
        print(
            "{",
        )
        for derivative in group:
            if not derivative.guard_template:
                continue
            template = lookup.get_template(derivative.guard_template)
            template_args = derivative.to_template_args()
            render_template(template, template_args, out)
        print(
            "\treport_missing_binop_case(instr, lhs, rhs);\n"
            "\tbreak;",
        )
        print(
            "}",
        )


def generate_declarations(derivatives, out):
    global print
    print = functools.partial(print, file=out)
    for derivative in derivatives:
        print(f"{derivative.signature()};\n")
        print(f"{derivative.slot_define()}\n")


parser = argparse.ArgumentParser()
parser.add_argument("-o", "--outfile", type=str, help="Path to the output file")
group = parser.add_mutually_exclusive_group()

group.add_argument(
    "-d",
    "--declarations",
    action="store_true",
    help="Generate slot definitions and forward declarations",
)

group.add_argument(
    "-c",
    "--binop-case-guards",
    action="store_true",
    help="Generate cases for the specialization switch",
)
parser.add_argument(
    "-s",
    "--cache-stats",
    action="store_true",
    default=False,
    help="Generate code to gather statistics on cache locality",
)

parser.add_argument(
    "-f",
    "--flatten-derivatives",
    action="store_true",
    default=os.environ.get("CMLQ_FLATTEN_ATTRIBUTE") in ("yes", "true", "t", "1"),
    help="Add the flatten attribute to derivative functions to force callee inlining",
)
args = parser.parse_args()

script_dir = os.path.dirname(__file__)
template_dir = os.path.join(script_dir, "cmlq_templates")
lookup = TemplateLookup(directories=[template_dir], strict_undefined=False)


with smart_open(args.outfile) as out:
    derivatives = build_derivatives(args.flatten_derivatives, args.cache_stats)

    if args.binop_case_guards:
        generate_case_guards(derivatives, lookup, out)
    elif args.declarations:
        generate_declarations(derivatives, out)
    else:
        generate_implementations(derivatives, lookup, out)
