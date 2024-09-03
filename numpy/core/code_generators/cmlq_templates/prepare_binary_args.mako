<%namespace file="cache_stats_macro.mako" import="*"/>

    PyObject *m1 = (*stack_pointer_ptr)[-2];
    PyObject *m2 = (*stack_pointer_ptr)[-1];

    %if commutative and right_scalar_name is not UNDEFINED:
    if (Py${right_scalar_name}_CheckExact(m1)) {
        PyObject *tmp = m1;
        m1 = m2;
        m2 = tmp;
    }
    %endif


    %if left_scalar_name is not UNDEFINED:
        if (NPY_UNLIKELY(!Py${left_scalar_name}_CheckExact(m1))) {
            <%count_stat("left_type_misses")%>
            goto deopt;
        }
        PyArray_Descr *left_descr = NULL;
        %if left_promotion is not None:
        left_descr = PyArray_DescrFromType(${left_promotion});
        %endif

        %if not cache_broadcast_array:
        PyArrayObject *lhs = (PyArrayObject *)PyArray_FromAny(m1, left_descr, 0, 0, 0, NULL);
        %else:
        assert(elem->state == BROADCAST && elem->result);
        PyArrayObject *lhs = elem->result;
        %endif
    %else:
        if (NPY_UNLIKELY(!PyArray_CheckExact(m1))) {
            <%count_stat("left_type_misses")%>
            goto deopt;
        }
        PyArrayObject *lhs = (PyArrayObject *)m1;
        if (NPY_UNLIKELY(PyArray_DESCR(lhs)->type_num != ${left_numpy_name})) {
            <%count_stat("left_type_misses")%>
            goto deopt;
        }
    %endif

    %if not right_scalar_name is UNDEFINED:
        if (NPY_UNLIKELY(!Py${right_scalar_name}_CheckExact(m2))) {
            <%count_stat("right_type_misses")%>
            goto deopt;
        }
        PyArray_Descr *right_descr = NULL;
        %if right_promotion is not None:
        right_descr = PyArray_DescrFromType(${right_promotion});
        %endif

        %if not cache_broadcast_array or left_scalar_name is not UNDEFINED:
        PyArrayObject *rhs = (PyArrayObject *)PyArray_FromAny(m2, right_descr, 0, 0, 0, NULL);
        %else:
        assert(elem->state == BROADCAST && elem->result);
        PyArrayObject *rhs = elem->result;
        %endif
    %else:
        if (NPY_UNLIKELY(!PyArray_CheckExact(m2))) {
            <%count_stat("right_type_misses")%>
            goto deopt;
        }
        PyArrayObject *rhs = (PyArrayObject *)m2;
        if (NPY_UNLIKELY(PyArray_DESCR(rhs)->type_num != ${right_numpy_name})) {
            <%count_stat("right_type_misses")%>
            goto deopt;
        }
    %endif

    // No need to check for ufunc overrides as we check for ndarray exact type
    // PyUFuncOverride_GetNonDefaultArrayUfunc even has a fast path for this case because there are no overrides

    PyArrayObject *ops[] = {lhs, rhs, NULL};
    PyArrayObject *result = NULL;

    // the non-scalar side determines the result shape
    %if left_numpy_name is not UNDEFINED:
    npy_intp result_ndims = PyArray_NDIM(lhs);
    npy_intp *result_shape = PyArray_SHAPE(lhs);
    %else:
    npy_intp result_ndims = PyArray_NDIM(rhs);
    npy_intp *result_shape = PyArray_SHAPE(rhs);
    %endif
