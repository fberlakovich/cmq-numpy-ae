${signature}
{
    CMLQ_PAPI_BEGIN("${opname}")
    %if locality_cache or locality_stats or cache_broadcast_array:
    <%include file="load_cache_elem.mako"/>
    %endif

    <%include file="prepare_binary_args.mako"/>

    %if locality_cache:
    <%include file="locality_cache.mako"/>
    %endif

    <%include file="array_op.mako" args="try_elide_temp=True"/>

deopt:
    %if locality_cache:
    elem = (CMLQLocalityCacheElem *)external_cache_pointer;
    if (elem->state != UNUSED) {

        if (elem->state == TRIVIAL) {
            Py_XDECREF(elem->result);
        } else if (elem->state == ITERATOR) {
            NpyIter_Deallocate(elem->iterator.cached_iter);
            elem->iterator.cached_iter = NULL;
        }

        elem->state = UNUSED;
        elem->result = NULL;

    }
    %endif
    return 2;

success:
%if cache_broadcast_array and left_scalar_name is not UNDEFINED:
    // the lhs is a cached broadcast array, no decref
%else:
    Py_DECREF(lhs);
%endif

%if cache_broadcast_array and left_scalar_name is UNDEFINED and right_scalar_name is not UNDEFINED:
    // the rhs is a cached broadcast array, no decref
%else:
    Py_DECREF(rhs);
%endif

%if right_scalar_name:
    Py_DECREF(m2);
%endif

    (*stack_pointer_ptr)--;
    assert(PyArray_CheckExact(result));
    (*stack_pointer_ptr)[-1] = (PyObject *)result;
    CMLQ_PAPI_END("${opname}")
    return 0;
fail:
    return -1;
}
