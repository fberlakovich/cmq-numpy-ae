<%inherit file="binop_case_guard.mako"/>
<%block name="install_handler">
    %if right_scalar_name == "Float":
        double exponent = PyFloat_AsDouble(rhs);
    %elif right_scalar_name == "Long":
        double exponent = PyLong_AsLong(rhs);
    %else:
        Unknown right scalar type
    %endif
        if (exponent == ${fixed_exponent}) {
            next_result_cache_index++;
            #ifdef CMLQ_STATS
            locality_cache[next_result_cache_index].stats.instr_ptr = instr;
            #endif
            specializer_info.SpecializeInstruction(instr, ${slot_name}, ${opname}, &locality_cache[next_result_cache_index]);
            return 1;
        }
</%block>
