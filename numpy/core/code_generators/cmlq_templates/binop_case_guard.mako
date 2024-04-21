    if((
    %if left_scalar_name is not UNDEFINED:
        Py${left_scalar_name}_CheckExact(lhs) &&
    %else:
        PyArray_CheckExact(lhs) &&
        PyArray_DESCR((PyArrayObject *)lhs)->type_num == ${left_numpy_name} &&
    %endif
    %if right_scalar_name is not UNDEFINED:
        Py${right_scalar_name}_CheckExact(rhs)
    %else:
        PyArray_CheckExact(rhs) &&
        PyArray_DESCR((PyArrayObject *)rhs)->type_num == ${right_numpy_name}
    %endif
       )
    %if commutative:
    || (
    %if left_scalar_name is not UNDEFINED:
        Py${left_scalar_name}_CheckExact(rhs) &&
    %else:
        PyArray_CheckExact(rhs) &&
        PyArray_DESCR((PyArrayObject *)rhs)->type_num == ${left_numpy_name} &&
    %endif
    %if right_scalar_name is not UNDEFINED:
        Py${right_scalar_name}_CheckExact(lhs)
    %else:
        PyArray_CheckExact(lhs) &&
        PyArray_DESCR((PyArrayObject *)lhs)->type_num == ${right_numpy_name}
    %endif
       )
    %endif
    )
    {
    <%block name="install_handler">
        next_result_cache_index++;
        #ifdef CMLQ_STATS
        locality_cache[next_result_cache_index].stats.instr_ptr = instr;
        #endif

        %if with_broadcast_cache_variant:

            PyArrayObject *op = NULL;
            PyArray_Descr *descr = NULL;
            int cache_possible = 0;

            %if left_scalar_name is not UNDEFINED:

                %if left_promotion is not None:
                descr = PyArray_DescrFromType(${left_promotion});
                %endif

                cache_possible = Py${left_scalar_name}_CheckExact(lhs) && specializer_info.IsOperandConstant(instr, *stack_pointer, 0);
                op = lhs;

                %if commutative:
                // try the commutative case
                if (!cache_possible) {
                    %if right_promotion is not None:
                    descr = PyArray_DescrFromType(${right_promotion});
                    %endif

                    cache_possible = Py${left_scalar_name}_CheckExact(rhs) && specializer_info.IsOperandConstant(instr, *stack_pointer, 1);
                    op = rhs;
                }
                %endif
            %endif

            %if right_scalar_name is not UNDEFINED and left_scalar_name is UNDEFINED:

                %if right_promotion is not None:
                descr = PyArray_DescrFromType(${right_promotion});
                %endif

                cache_possible = Py${right_scalar_name}_CheckExact(rhs) && specializer_info.IsOperandConstant(instr, *stack_pointer, 1);
                op = rhs;

                %if commutative:
                // try the commutative case
                if (!cache_possible) {
                    %if left_promotion is not None:
                    descr = PyArray_DescrFromType(${left_promotion});
                    %endif

                    cache_possible = Py${right_scalar_name}_CheckExact(lhs) && specializer_info.IsOperandConstant(instr, *stack_pointer, 0);
                    op = lhs;
                }
                %endif
            %endif

            if (cache_possible) {
                // precompute the broadcast array
                locality_cache[next_result_cache_index].state = BROADCAST;
                locality_cache[next_result_cache_index].result = (PyArrayObject *)PyArray_FromAny(op, descr, 0, 0, 0, NULL);
                specializer_info.SpecializeInstruction(instr, ${slot_name}_BROADCAST_CACHE, ${opname}_broadcast_cache, &locality_cache[next_result_cache_index]);
            } else {
                specializer_info.SpecializeInstruction(instr, ${slot_name}, ${opname}, &locality_cache[next_result_cache_index]);
            }

        %else:
            specializer_info.SpecializeInstruction(instr, ${slot_name}, ${opname}, &locality_cache[next_result_cache_index]);
        %endif
        return 1;
    </%block>
    }