from cython_gsl cimport *

def t_gsl_sort_smallest():
    cdef double a[5], b[2]
    a1 = [3, 4, 1, 5, 2]
    for i from 0 <= i < 5:
        a[i] = a1[i] + 0.1
    gsl_sort_smallest(b, 2, a, 1, 5)
    ary = [b[0] - 1.1, b[1] - 2.1]
    gsl_sort_largest(b, 2, a, 1, 5)
    ary.extend([b[0] - 5.1, b[1] - 4.1])
    return ary

def t_gsl_sort_smallest_index():
    cdef double a[5]
    cdef size_t b[2]
    a1 = [3, 4, 1, 5, 2]
    for i from 0 <= i < 5:
        a[i] = a1[i] + 0.1
    gsl_sort_smallest_index(b, 2, a, 1, 5)
    ary = [b[0] - 2, b[1] - 4]
    gsl_sort_largest_index(b, 2, a, 1, 5)
    ary.extend([b[0] - 3, b[1] - 1])
    return ary

def t_gsl_sort_vector_smallest():
    cdef gsl_vector * a
    a = gsl_vector_alloc(5)
    cdef double b[2]
    a1 = [3, 4, 1, 5, 2]
    for i from 0 <= i < 5:
        gsl_vector_set(a, i, a1[i] + 0.1)
    gsl_sort_vector_smallest(b, 2, a)
    ary = [b[0] - 1.1, b[1] - 2.1]
    gsl_sort_vector_largest(b, 2, a)
    ary.extend([b[0] - 5.1, b[1] - 4.1])
    return ary

def t_gsl_sort_vector_smallest_index():
    cdef gsl_vector * a
    a = gsl_vector_alloc(5)
    cdef size_t b[2]
    a1 = [3, 4, 1, 5, 2]
    for i from 0 <= i < 5:
        gsl_vector_set(a, i, a1[i] + 0.1)
    gsl_sort_vector_smallest_index(b, 2, a)
    ary = [b[0] - 2, b[1] - 4]
    gsl_sort_vector_largest_index(b, 2, a)
    ary.extend([b[0] - 3, b[1] - 1])
    return ary

def t_gsl_sort():
    cdef double a[5]
    a1 = [3, 4, 1, 5, 2]
    for i from 0 <= i < 5:
        a[i] = a1[i] + 0.1
    gsl_sort(a, 1, 5)
    ary = [a[0] - 1.1, a[1] - 2.1, a[2] - 3.1, a[3] - 4.1, a[4] - 5.1]
    return ary

def t_gsl_sort_index():
    cdef double a[5]
    cdef size_t b[5]
    a1 = [3, 4, 1, 5, 2]
    for i from 0 <= i < 5:
        a[i] = a1[i] + 0.1
    gsl_sort_index(b, a, 1, 5)
    ary = [b[0] -2, b[1] - 4, b[2], b[3] - 1, b[4] - 3]
    return ary

def t_gsl_sort_vector():
    cdef gsl_vector * a
    a = gsl_vector_alloc(5)
    a1 = [3, 4, 1, 5, 2]
    for i from 0 <= i < 5:
        gsl_vector_set(a, i, a1[i] + 0.1)
    gsl_sort_vector(a)
    ary = []
    for i from 0 <= i < 5:
        ary.append(gsl_vector_get(a,i) - i - 1.1)
    return ary

def t_gsl_sort_vector_index():
    cdef gsl_vector * a
    a = gsl_vector_alloc(5)
    cdef gsl_permutation * p
    p = gsl_permutation_alloc(5)
    a1 = [3, 4, 1, 5, 2]
    for i from 0 <= i < 5:
        gsl_vector_set(a, i, a1[i] + 0.1)
    gsl_sort_vector_index(p, a)
    ary = []
    cdef int j
    for i from 0 <= i < 5:
        j = gsl_permutation_get(p, i)
        ary.append(gsl_vector_get(a,j) - i - 1.1)
    return ary


cdef int cmp1(void * a, void * b):
    cdef int a1, b1
    a1 = (<int *> a)[0]
    a1 = (a1 * a1)%10
    b1 = (<int *> b)[0]
    b1 = (b1 * b1)%10
    if (a1 < b1):
        return -1
    elif (a1 > b1):
        return 1
    else:
        return 0

def t_gsl_heapsort():
    cdef int a[5], b[2]
    cdef sizeof_int
    sizeof_int = 4
    a1 = [3, 4, 1, 5, 2]
    for i from 0 <= i < 5:
        a[i] = a1[i]
    gsl_heapsort(<void *> a, 5, sizeof_int,  <gsl_comparison_fn_t>cmp1)
    r = [1,2,5,4,3]
    ary = []
    for i from 0 <= i < 5:
        ary.append(a[i] - r[i])
    return ary

def t_gsl_heapsort_index():
    cdef size_t b[5]
    cdef int a[5]
    cdef sizeof_int
    sizeof_int = 4
    a1 = [3, 4, 1, 5, 2]
    for i from 0 <= i < 5:
        a[i] = a1[i]
    gsl_heapsort_index(b, <void *> a, 5, sizeof_int, <gsl_comparison_fn_t>cmp1)
    ary = [b[0] -2, b[1] - 4, b[2] - 3, b[3] - 1, b[4]]
    return ary
