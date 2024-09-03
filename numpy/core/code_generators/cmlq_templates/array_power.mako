<%namespace file="cache_stats_macro.mako" import="*"/>

${signature}
{
    %if locality_cache or locality_stats:
    <%include file="load_cache_elem.mako"/>
    %endif

    <%include file="prepare_binary_args.mako"/>

<%def name="auxdata_init()">
%if left_numpy_name != "NPY_DOUBLE":
    auxdata = npy_powf;
%endif
</%def>

%if right_scalar_name == "Long":
    long exponent = PyLong_AsLong(m2);
%elif right_scalar_name == "Float":
    double exponent = PyFloat_AsDouble(m2);
%else:
    // cause a compile error
    Unknown scalar type for exponent
%endif

%if fixed_exponent is None:
    %if locality_cache:
    <%include file="locality_cache.mako", args="arity=2"/>
    %endif
    <%include file="array_op.mako" args="try_elide_temp=False, auxdata_init=auxdata_init"/>
%else:
    %if locality_cache:
    <%include file="locality_cache.mako", args="auxdata_init=auxdata_init, arity=1"/>
    %endif
    if (exponent != ${fixed_exponent}) {
        <%count_stat("exponent_type_misses")%>
        goto deopt;
    }
    ops[1] = NULL;
    <%include file="array_op.mako" args="try_elide_temp=True, auxdata_init=auxdata_init, arity=1"/>
%endif

deopt:
    return 2;

success:
    Py_DECREF(lhs);
    Py_DECREF(rhs);

    %if right_scalar_name:
        Py_DECREF(m2);
    %endif

    (*stack_pointer_ptr)--;
    assert(PyArray_CheckExact(result));
    (*stack_pointer_ptr)[-1] = (PyObject *)result;
    return 0;
fail:
    return -1;
}
