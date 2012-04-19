from cython_gsl cimport *


def t_gsl_permutation_alloc():
    cdef gsl_permutation * p
    p = gsl_permutation_alloc (3)
    gsl_permutation_init (p)
    cdef size_t size1
    size1 = gsl_permutation_size(p)
    ary = []
    ary.append(size1 - 3)
    cdef size_t * ai
    ai = gsl_permutation_data(p)
    ary.extend([ai[0], ai[1] - 1, ai[2] - 2])
    gsl_permutation_free(p)
    return ary


def t_gsl_permutation_calloc():
    cdef gsl_permutation * p
    p = gsl_permutation_calloc (3)
    ary = []
    cdef size_t * ai
    ai = gsl_permutation_data(p)
    ary.extend([ai[0], ai[1] - 1, ai[2] - 2])
    gsl_permutation_free(p)
    return ary

def t_gsl_permutation_get():
    cdef gsl_permutation * p
    p = gsl_permutation_calloc (3)
    ary = []
    for i from 0 <= i < 3:
        ary.append(gsl_permutation_get(p, i) - i)
    gsl_permutation_free(p)
    return ary

def t_gsl_permutation_next():
    cdef gsl_permutation * p
    p = gsl_permutation_calloc (3)
    gsl_permutation_next(p)
    ary = [0, 2, 1]
    for i from 0 <= i < 3:
        ary[i] = ary[i] - gsl_permutation_get(p, i)
    gsl_permutation_free(p)
    return ary

def t_gsl_permutation_prev():
    cdef gsl_permutation * p
    p = gsl_permutation_calloc (3)
    gsl_permutation_next(p)
    gsl_permutation_next(p)
    gsl_permutation_prev(p)
    ary = [0, 2, 1]
    for i from 0 <= i < 3:
        ary[i] = ary[i] - gsl_permutation_get(p, i)
    gsl_permutation_free(p)
    return ary

def t_gsl_permutation_valid():
    cdef gsl_permutation * p
    p = gsl_permutation_calloc (3)
    p.data[2] = 1; p.data[1] = 2
    ary = []
    # gsl_permutation_valid(p) == 0 if p is valid
    ary.append(gsl_permutation_valid(p))
    gsl_permutation_prev(p)
    cdef size_t * ai
    ai = gsl_permutation_data(p)
    ary.extend([ai[0], ai[1] - 1, ai[2] - 2])
    gsl_permutation_free(p)
    return ary

def t_gsl_permutation_swap():
    cdef gsl_permutation * p
    p = gsl_permutation_calloc (3)
    gsl_permutation_swap(p, 0, 2)
    cdef size_t * ai
    ai = gsl_permutation_data(p)
    ary = [ai[0] -2, ai[1] - 1, ai[2]]
    gsl_permutation_free(p)
    return ary

def t_gsl_permutation_memcpy():
    cdef gsl_permutation * p1, * p2
    p1 = gsl_permutation_calloc (3)
    p2 = gsl_permutation_alloc (3)
    gsl_permutation_swap(p1, 0, 2)
    gsl_permutation_memcpy(p2, p1)
    cdef size_t * ai
    ai = gsl_permutation_data(p2)
    ary = [ai[0] -2, ai[1] - 1, ai[2]]
    gsl_permutation_free(p1)
    gsl_permutation_free(p2)
    return ary

def t_gsl_permutation_reverse():
    cdef gsl_permutation * p
    p = gsl_permutation_alloc (5)
    a = [2,4,1,3,0]
    cdef int i
    for i from 0 <= i <= 4:
        p.data[i] = a[i]
    cdef size_t * ai
    gsl_permutation_reverse(p)
    ai = gsl_permutation_data(p)
    ary = []
    a.reverse()
    for i from 0 <= i <= 4:
        ary.append(ai[i] - a[i])
    gsl_permutation_free(p)
    return ary

def t_gsl_permutation_inverse():
    cdef gsl_permutation * p1, * p2, * p3
    p1 = gsl_permutation_alloc (5)
    p2 = gsl_permutation_alloc (5)
    p3 = gsl_permutation_alloc (5)
    a = [0, 3, 1, 4, 2]
    cdef int i
    for i from 0 <= i <= 4:
        p1.data[i] = a[i]
    cdef size_t * ai
    gsl_permutation_inverse(p2, p1)
    ai = gsl_permutation_data(p2)
    ary = []
    b = [0, 2, 4, 1, 3]
    for i from 0 <= i <= 4:
        ary.append(ai[i] - b[i])
    # p3 = p1 * p2
    gsl_permutation_mul(p3, p1, p2)
    # p3 == identity permutation
    ai = gsl_permutation_data(p3)
    for i from 0 <= i <= 4:
        ary.append(ai[i] - i)
    gsl_permutation_free(p1)
    gsl_permutation_free(p2)
    gsl_permutation_free(p3)
    return ary

def t_gsl_permute_vector():
    cdef gsl_permutation * p1
    cdef gsl_vector * v1
    p1 = gsl_permutation_calloc (4)
    v1 = gsl_vector_alloc(4)
    for i from 0 <= i < 4:
        gsl_vector_set(v1, i, i + 1.1)
    gsl_permutation_swap(p1, 2,3)
    gsl_permute_vector(p1, v1)
    ary = []
    a = [1.1,2.1,4.1,3.1]
    for i from 0 <= i < 4:
        ary.append(gsl_vector_get(v1, i) - a[i])
    gsl_permutation_free(p1)
    gsl_vector_free(v1)
    return ary

def t_gsl_permute_vector_inverse():
    cdef gsl_permutation * p1
    cdef gsl_vector * v1, * v2
    p1 = gsl_permutation_calloc (4)
    v1 = gsl_vector_alloc(4)
    v2 = gsl_vector_alloc(4)
    for i from 0 <= i < 4:
        gsl_vector_set(v1, i, i + 1.1)
        gsl_vector_set(v2, i, i + 1.1)
    gsl_permutation_swap(p1, 2,3)
    gsl_permute_vector(p1, v2)
    gsl_permute_vector_inverse(p1, v2)
    ary = []
    for i from 0 <= i < 4:
        ary.append(gsl_vector_get(v1, i) - gsl_vector_get(v2, i))
    gsl_permutation_free(p1)
    gsl_vector_free(v1)
    gsl_vector_free(v2)
    return ary

def t_gsl_permute_vector_complex():
    cdef gsl_permutation * p1
    cdef gsl_vector_complex * v1
    cdef gsl_vector_complex * v2
    cdef gsl_complex z
    p1 = gsl_permutation_calloc (4)
    v1 = gsl_vector_complex_alloc(4)
    v2 = gsl_vector_complex_alloc(4)
    for i from 0 <= i < 4:
        gsl_vector_complex_set(v1, i, gsl_complex_rect(i + 1.1, i + 1.2))
        gsl_vector_complex_set(v2, i, gsl_complex_rect(i + 1.1, i + 1.2))
    # swap manually
    gsl_vector_complex_set(v2, 2, gsl_complex_rect(3 + 1.1, 3 + 1.2))
    gsl_vector_complex_set(v2, 3, gsl_complex_rect(2 + 1.1, 2 + 1.2))
    gsl_permutation_swap(p1, 2,3)
    gsl_permute_vector_complex(p1, v1)
    ary = []
    for i from 0 <= i < 4:
        z = gsl_complex_sub(gsl_vector_complex_get(v1, i), gsl_vector_complex_get(v2, i))
        ary.extend([GSL_REAL(z), GSL_IMAG(z)])
    gsl_permutation_free(p1)
    gsl_vector_complex_free(v1)
    gsl_vector_complex_free(v2)
    return ary

def t_gsl_permute_vector_complex_inverse():
    cdef gsl_permutation * p1
    cdef gsl_vector_complex * v1
    cdef gsl_vector_complex * v2
    cdef gsl_complex z
    p1 = gsl_permutation_calloc (4)
    v1 = gsl_vector_complex_alloc(4)
    v2 = gsl_vector_complex_alloc(4)
    for i from 0 <= i < 4:
        gsl_vector_complex_set(v1, i, gsl_complex_rect(i + 1.1, i + 1.2))
        gsl_vector_complex_set(v2, i, gsl_complex_rect(i + 1.1, i + 1.2))
    gsl_permutation_swap(p1, 2,3)
    gsl_permute_vector_complex(p1, v2)
    gsl_permute_vector_complex(p1, v2)
    ary = []
    for i from 0 <= i < 4:
        z = gsl_complex_sub(gsl_vector_complex_get(v1, i), gsl_vector_complex_get(v2, i))
        ary.extend([GSL_REAL(z), GSL_IMAG(z)])
    gsl_permutation_free(p1)
    gsl_vector_complex_free(v1)
    gsl_vector_complex_free(v2)
    return ary

def t_gsl_permutation_linear_to_canonical():
    cdef gsl_permutation * p1, * p2, * p3
    p1 = gsl_permutation_alloc (5)
    p2 = gsl_permutation_alloc (5)
    p3 = gsl_permutation_alloc (5)
    a = [2, 4, 3, 0, 1]
    for i from 0 <= i < 5:
        p1.data[i] = a[i]
    gsl_permutation_linear_to_canonical(p2, p1)
    b = [1, 4, 0, 2, 3]
    ary = []
    for i from 0 <= i < 5:
        ary.append(p2.data[i] - b[i])
    gsl_permutation_canonical_to_linear(p3, p2)
    for i from 0 <= i < 5:
        ary.append(p3.data[i] - p1.data[i])
    gsl_permutation_free(p1)
    gsl_permutation_free(p2)
    gsl_permutation_free(p3)
    return ary

def t_gsl_permutation_fprintf():
    cdef gsl_permutation * p1, * p2
    p1 = gsl_permutation_calloc (3)
    p2 = gsl_permutation_calloc (3)
    cdef FILE * f1
    f1 = fopen ("test.dat", "w")
    gsl_permutation_fprintf (f1, p1, " %u")
    fclose(f1)
    cdef FILE * f2
    f2 = fopen ("test.dat", "r")
    gsl_permutation_fscanf(f2, p2)
    fclose(f2)
    ary = []
    for i from 0 <= i < 3:
        ary.append(p2.data[i] - i)
    gsl_permutation_free(p1)
    gsl_permutation_free(p2)
    return ary

def t_gsl_permutation_linear_cycles():
    cdef gsl_permutation * p1, * p2, * p3
    p1 = gsl_permutation_alloc (5)
    p2 = gsl_permutation_alloc (5)
    a = [2, 4, 3, 0, 1]
    for i from 0 <= i < 5:
        p1.data[i] = a[i]
    gsl_permutation_linear_to_canonical(p2, p1)
    b = [1, 4, 0, 2, 3]
    ary = []
    for i from 0 <= i < 5:
        ary.append(p2.data[i] - b[i])
    cdef size_t c1, c2
    c1 = gsl_permutation_linear_cycles(p1)
    c2 = gsl_permutation_canonical_cycles(p2)
    ary.extend([c1 - 2, c2 - 2])
    gsl_permutation_free(p1)
    gsl_permutation_free(p2)
    return ary
