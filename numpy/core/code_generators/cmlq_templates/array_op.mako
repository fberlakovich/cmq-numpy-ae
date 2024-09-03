<%page args="result_type, try_elide_temp=False, auxdata_init=None, arity=2, inplace=False"/>
<% assert arity == 2 or arity == 1 %>
<%namespace file="cache_stats_macro.mako" import="*"/>

    npy_intp fixed_strides[${arity + 1}];
    int fast_path = 1;
    %if left_numpy_name is not UNDEFINED and right_numpy_name is not UNDEFINED and arity == 2:

    if (PyArray_NDIM(lhs) != PyArray_NDIM(rhs) ||
        !PyArray_CompareLists(PyArray_SHAPE(lhs), PyArray_SHAPE(rhs), PyArray_NDIM(lhs))) {
        fast_path = 2;
    } else {
        if (PyArray_NDIM(lhs) == 1) {
            fixed_strides[0] = PyArray_STRIDES(lhs)[0];
            fixed_strides[1] = PyArray_STRIDES(rhs)[0];
            %if inplace:
            fixed_strides[2] = PyArray_STRIDES(lhs)[0];
            %endif
        } else {
            if (!(PyArray_FLAGS(lhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS)) ||
                !(PyArray_FLAGS(rhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS))) {
                fast_path = 2;
            } else {
                fixed_strides[0] = PyArray_ITEMSIZE(lhs);
                fixed_strides[1] = PyArray_ITEMSIZE(rhs);
                %if inplace:
                fixed_strides[2] = PyArray_ITEMSIZE(lhs);
                %endif
            }
        }
    }

    %else:
    %if left_numpy_name is not UNDEFINED:

    if (PyArray_NDIM(lhs) == 1) {
        fixed_strides[0] = PyArray_STRIDES(lhs)[0];
    } else {
        if (!(PyArray_FLAGS(lhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS))) {
            fast_path = 2;
        } else {
            fixed_strides[0] = PyArray_ITEMSIZE(lhs);
        }
    }

    %if arity == 2:
    fixed_strides[1] = 0;
    %endif

    %elif right_numpy_name is not UNDEFINED:

    if (PyArray_NDIM(rhs) == 1) {
        fixed_strides[1] = PyArray_STRIDES(rhs)[0];
    } else {
        if (!(PyArray_FLAGS(rhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS))) {
            fast_path = 2;
        } else {
            fixed_strides[1] = PyArray_ITEMSIZE(rhs);
        }
    }

    %if arity == 2:
    fixed_strides[0] = 0;
    %endif

    %endif
    %endif

    %if not inplace:
        %if try_elide_temp:
            // try to avoid creating a temporary array
            %if arity == 2:
            determine_elide_temp_binary(m1, m2, (PyObject **)&result, ${1 if commutative else 0});
            %elif arity == 1:
            if (can_elide_temp_unary(m1)) {
                result = m1;
            }
            %endif
        %endif
    %else:
        // inplace operation, the result is the same as the lhs
        Py_INCREF(lhs);
        result = lhs;
    %endif

    <%count_stat("temp_elision_hits", "result != NULL")%>

    if(fast_path == 1) {
        <%count_stat("trivial_case")%>
        %if not inplace:
            %if try_elide_temp:
            if (result == NULL) {
            %endif
                // there was no temp elision so we need to create a new array
                PyArray_Descr *result_descr = PyArray_DescrFromType(${result_type});

                // at this time we know that the arrays can't have different dimensions otherwise we would have taken the iterator path
                result = (PyArrayObject *)PyArray_NewFromDescr(
                        &PyArray_Type, result_descr, result_ndims,
                        result_shape, NULL, NULL,
                        0, NULL);
                fixed_strides[${arity}] = PyArray_DESCR(result)->elsize;
            %if try_elide_temp:
            } else {
                // there was temp elision
                Py_INCREF(result);

                // copy the strides from the lhs (because it is the result as well)
                if (PyArray_NDIM(lhs) == 1) {
                    fixed_strides[${arity}] = PyArray_STRIDES(lhs)[0];
                } else {
                    fixed_strides[${arity}] = PyArray_ITEMSIZE(lhs);
                }
            }
            %endif
        %else:
        assert(lhs == result);
        %endif

        npy_intp count = PyArray_MultiplyList(result_shape, result_ndims);

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
        NPY_BEGIN_THREADS_THRESHOLDED(count);

        NpyAuxData *auxdata = NULL;
        %if auxdata_init is not None:
        ${auxdata_init()}
        %endif

        CMLQ_PAPI_REGION("core_loop", ${loop_function}(data, &count, fixed_strides, auxdata));

        NPY_END_THREADS;

        %if locality_cache and not inplace:
            %if locality_cache_size_limit is not UNDEFINED:
            if (PyArray_NBYTES(result) >= ${locality_cache_size_limit}) {
                <%count_stat("result_too_big")%>
                cache_miss(elem);
                elem->state = DISABLED;
            }
            %endif

        if (elem->state != DISABLED && result != lhs) {
            if (elem->miss_counter >= 0) {
                elem->result = result;
                ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                Py_INCREF(elem->result);

                elem->trivial.count = count;
                elem->trivial.fixed_strides[0] = fixed_strides[0];
                elem->trivial.fixed_strides[1] = fixed_strides[1];
                elem->trivial.fixed_strides[2] = fixed_strides[2];
                elem->state = TRIVIAL;
                <%count_stat("trivial_cache_init")%>
            } else {
                // warm up the result cache
                elem->miss_counter++;
            }
        }
        %endif

        %if inplace:
        // this is an inplace operation. We do not cache the result here because no result array is allocated anyway
        %endif


        goto success;
    }

    if (fast_path == 2) {
        <%count_stat("iterator_case")%>
        // use the standard flags here
        npy_uint32 iter_flags = NPY_ITER_EXTERNAL_LOOP | NPY_ITER_REFS_OK |
                                NPY_ITER_ZEROSIZE_OK | NPY_ITER_BUFFERED |
                                NPY_ITER_GROWINNER | NPY_ITER_DELAY_BUFALLOC |
                                NPY_ITER_COPY_IF_OVERLAP;

        PyArray_Descr *result_descr = PyArray_DescrFromType(${result_type});

        %if arity == 2:
        PyArray_Descr *descriptors[3] = {PyArray_DESCR(lhs), PyArray_DESCR(rhs), result_descr};
        npy_uint32 op_flags[3] = {NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_ITER_WRITEONLY | NPY_UFUNC_DEFAULT_OUTPUT_FLAGS};

        // if result is not NULL, it means we need to reuse it, if it is NULL the iterator will create a new array
        ops[2] = result;
        %elif arity == 1:
        PyArray_Descr *descriptors[2] = {PyArray_DESCR(lhs), result_descr};
        npy_uint32 op_flags[2] = {NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_ITER_WRITEONLY | NPY_UFUNC_DEFAULT_OUTPUT_FLAGS};

        // if result is not NULL, it means we need to reuse it, if it is NULL the iterator will create a new array
        ops[1] = result;
        %endif

        %if inplace:
        assert(ops[0] == ops[${arity}]);
        %endif

        NpyIter *iter = NpyIter_AdvancedNew(${arity + 1}, ops,
                                            iter_flags,
                                            NPY_KEEPORDER, NPY_UNSAFE_CASTING,
                                            op_flags, descriptors,
                                            -1, NULL, NULL, NPY_BUFSIZE);
        if (iter == NULL) {
            goto fail;
        }

        /* Set the output array as output (the iterator might have created an array) */
        PyArrayObject **op_it = NpyIter_GetOperandArray(iter);
        result = op_it[${arity}];

        // the result will be pushed to the stack
        Py_INCREF(result);

        /* Only do the loop if the iteration size is non-zero */
        npy_intp full_size = NpyIter_GetIterSize(iter);
        if (full_size == 0) {
            Py_DECREF(result_descr);
            if (!NpyIter_Deallocate(iter)) {
                goto fail;
            }
            goto success;
        }

        char *baseptrs[${arity + 1}];

        baseptrs[0] = PyArray_BYTES(op_it[0]);
        baseptrs[1] = PyArray_BYTES(op_it[1]);
        %if arity == 2:
        baseptrs[2] = PyArray_BYTES(op_it[2]);
        %endif

        /* Get the variables needed for the loop */
        NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }
        char **dataptr = NpyIter_GetDataPtrArray(iter);
        npy_intp *strides = NpyIter_GetInnerStrideArray(iter);
        npy_intp *countptr = NpyIter_GetInnerLoopSizePtr(iter);

        NpyAuxData *auxdata = NULL;
        %if auxdata_init is not None:
        ${auxdata_init()}
        %endif

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(full_size);

        /* The reset may copy the first buffer chunk, which could cause FPEs */
        if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }

        /* Execute the loop */
        do {
            CMLQ_PAPI_REGION("core_loop", ${loop_function}(dataptr, countptr, strides, auxdata));
        } while (iternext(iter));

        NPY_END_THREADS;


        %if locality_cache and not inplace:
        int should_deallocate = 0;

        %if locality_cache_size_limit is not UNDEFINED:
        if (PyArray_NBYTES(result) >= ${locality_cache_size_limit}) {
            <%count_stat("result_too_big")%>
            cache_miss(elem);
            elem->state = DISABLED;
        }
        %endif

        if (elem->state != DISABLED && result != lhs) {
            if (elem->miss_counter >= 0) {
                elem->state = ITERATOR;

                elem->iterator.countptr = countptr;
                elem->iterator.dataptr = dataptr;
                elem->iterator.strides = strides;

                // we do not need to increase the refcnt here because the iterator holds the reference
                elem->result = result;
                ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                elem->iterator.cached_iter = iter;
                elem->iterator.iter_next = *iternext;
                <%count_stat("iterator_cache_init")%>
            } else {
                // warm up the result cache
                elem->miss_counter++;
                should_deallocate = 1;
            }
        } else {
            // the iterator is not cached, so we need to deallocate it
            should_deallocate = 1;
        }
        %else:
        // no locality cache, we always need to deallocate the iterator
        %if inplace:
        // this is an inplace operation. We do not cache the result here because no result array is allocated anyway
        %endif

        int should_deallocate = 1;
        %endif

        if (should_deallocate) {
            if (!NpyIter_Deallocate(iter)) {
                 goto fail;
            }
        }

        Py_DECREF(result_descr);
        goto success;
    }

    // some other error we can't handle
    raise(SIGTRAP);
