<%page args="arity=2, auxdata_init=None"/>
<% assert arity == 2 or arity == 1 %>

<%namespace file="cache_stats_macro.mako" import="*"/>


// we can only use the result cache if we are the sole owner of the result object
// and if the object properties match the required result properties
#define RESULT_CACHE_VALID(elem) ${"\\"}
    ((PyObject *)(elem->result))->ob_refcnt == 1 && ${"\\"}
    PyArray_NDIM(elem->result) == result_ndims && ${"\\"}
    PyArray_CompareLists(result_shape, PyArray_SHAPE(elem->result), result_ndims)

#define CACHE_MATCH_TRIVIAL() ${"\\"}
    elem->state == TRIVIAL

#define CACHE_MATCH_ITERATOR() ${"\\"}
    elem->state == ITERATOR

    if (CACHE_MATCH_TRIVIAL()) {
        <%count_stat("trivial_cache_hits")%>

        %if not inplace:
        if (RESULT_CACHE_VALID(elem)) {
        %endif
            <%count_stat("result_cache_hits")%>

            %if inplace:
            result = lhs;
            %else:
            result = elem->result;
            %endif

            // in addition to the cache, the result will be pushed to the stack
            Py_INCREF(result);

            char *data[${arity + 1}];
            %if arity == 2:
            data[0] = PyArray_BYTES(lhs);
            data[1] = PyArray_BYTES(rhs);
            data[2] = PyArray_BYTES(result);
            %elif arity == 1:
            data[0] = PyArray_BYTES(lhs);
            data[1] = PyArray_BYTES(result);
            %endif

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(elem->trivial.count);

            NpyAuxData *auxdata = NULL;
            ${loop_function}(data, &elem->trivial.count, elem->trivial.fixed_strides, auxdata);

            NPY_END_THREADS;

            goto success;
        %if not inplace:
        }
        else {
            <%count_stat("result_cache_misses")%>
            <%count_stat("refcnt_misses", "((PyObject *)(elem->result))->ob_refcnt != 1")%>
            <%count_stat("ndims_misses", "PyArray_NDIM(elem->result) != result_ndims")%>
            <%count_stat("shape_misses", "!PyArray_CompareLists(result_shape, PyArray_SHAPE(elem->result), result_ndims)")%>
            trivial_cache_miss(elem);
        }
        %endif
    } else if (CACHE_MATCH_ITERATOR()) {
        <%count_stat("iterator_cache_hits")%>

        %if not inplace:
        if (RESULT_CACHE_VALID(elem)) {
        %endif
            <%count_stat("result_cache_hits")%>

            %if inplace:
            result = lhs;
            %else:
            result = elem->result;
            %endif

            // the result will be pushed to the stack
            Py_INCREF(result);

            NpyIter *iter = elem->iterator.cached_iter;

            PyArrayObject *op_it[${arity + 1}];
            op_it[0] = lhs; // is always LHS

            %if arity == 2:
            op_it[1] = rhs;
            %endif

            op_it[${arity}] = result;

            // can reuse the old one, nothing changes there...
            NpyIter_IterNextFunc *iter_next = elem->iterator.iter_next;
            char **dataptr = elem->iterator.dataptr;
            npy_intp *strides = elem->iterator.strides;

            char *baseptrs[3];
            baseptrs[0] = PyArray_BYTES(op_it[0]);
            baseptrs[1] = PyArray_BYTES(op_it[1]);
            %if arity == 2:
            baseptrs[2] = PyArray_BYTES(op_it[2]);
            %endif

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(NpyIter_GetIterSize(elem->iterator.cached_iter));

            /* The reset may copy the first buffer chunk, which could cause FPEs */
            if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
                NpyIter_Deallocate(iter);
                goto fail;
            }

            NpyAuxData *auxdata = NULL;
            %if auxdata_init is not None:
            ${auxdata_init()}
            %endif

            /* Execute the loop */
            do {
                ${loop_function}(dataptr, elem->iterator.countptr, strides, auxdata);
            } while (iter_next(iter));

            NPY_END_THREADS;

            // standard epilogue here...
            goto success;
       %if not inplace:
       } else {
            <%count_stat("result_cache_misses")%>
            <%count_stat("refcnt_misses", "((PyObject *)(elem->result))->ob_refcnt != 1")%>
            <%count_stat("ndims_misses", "PyArray_NDIM(elem->result) != result_ndims")%>
            <%count_stat("shape_misses", "!PyArray_CompareLists(result_shape, PyArray_SHAPE(elem->result), result_ndims)")%>

            iterator_cache_miss(elem);
       }
       %endif
    } else {
        assert(elem->state == UNUSED || elem->state == DISABLED);

        if (elem->state == TRIVIAL) {
            // cache collision, free the old result element
            trivial_cache_miss(elem);
            <%count_stat("trivial_cache_collisions")%>
        }

        if (elem->state == ITERATOR) {
            iterator_cache_miss(elem);
            <%count_stat("iterator_cache_collisions")%>
        }

        // initialize cache element
        if (elem->state == UNUSED) {
            elem->result = NULL;
        }
    }
