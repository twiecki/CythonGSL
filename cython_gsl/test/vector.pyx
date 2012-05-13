from cython_gsl cimport *

def t_gsl_block():
    cdef gsl_block * b
    b = gsl_block_alloc(100)
    cdef size_t size1, size2
    size1 = b.size
    size2 = gsl_block_size(b)
    gsl_block_free(b)
    return (size1 - 100, size2 - 100)

def t_gsl_vector_set():
    cdef gsl_vector *v
    v = gsl_vector_alloc(3)
    cdef int i
    for i from 0 <=i < 3:
        gsl_vector_set(v, i, i + 0.1)
    a = []
    for i from 0 <=i < 3:
        a.append(gsl_vector_get(v, i) - i - 0.1)
    gsl_vector_free(v)
    return a

def t_gsl_vector_fprintf():
    cdef int i
    cdef gsl_vector *v
    v = gsl_vector_alloc(100)

    for i from 0 <= i < 100:
        gsl_vector_set (v, i, 1.23 + i)

    cdef FILE * f
    f = fopen ("test.dat", "w")
    gsl_vector_fprintf (f, v, "%.5g")
    fclose (f)

    cdef gsl_vector *v2
    v2 = gsl_vector_alloc(10)


    cdef FILE * f2
    f2 = fopen ("test.dat", "r")
    gsl_vector_fscanf (f2, v2)
    fclose (f2)

    s1 = ""
    for i from 0 <= i < 10:
        s1 =  s1 + "%g\n" % gsl_vector_get(v2, i)

    s2 = '''1.23
  2.23
  3.23
  4.23
  5.23
  6.23
  7.23
  8.23
  9.23
  10.23
  '''
    gsl_vector_free(v)
    gsl_vector_free(v2)
    return s1 == s2

def t_gsl_vector_set_zero():
    cdef gsl_vector *v
    v = gsl_vector_alloc(3)
    cdef int i
    for i from 0 <=i < 3:
        gsl_vector_set(v, i, i + 0.1)
    gsl_vector_set_zero(v)
    a = []
    for i from 0 <=i < 3:
        a.append(gsl_vector_get(v, i))
    gsl_vector_free(v)
    return a

def t_gsl_vector_set_all():
    cdef gsl_vector *v
    v = gsl_vector_alloc(3)
    cdef int i
    gsl_vector_set_all(v, 1.4)
    a = []
    for i from 0 <=i < 3:
        a.append(gsl_vector_get(v, i) - 1.4)
    gsl_vector_free(v)
    return a

def t_gsl_vector_set_basis():
    cdef gsl_vector *v
    v = gsl_vector_alloc(3)
    gsl_vector_set_basis(v, 1)
    #gsl_vector_free(v)
    a = [gsl_vector_get(v, 0),gsl_vector_get(v, 1) -1, gsl_vector_get(v, 2)]
    gsl_vector_free(v)
    return a

def t_gsl_vector_calloc():
    cdef gsl_vector *v
    v = gsl_vector_calloc(3)
    cdef int i
    a = []
    for i from 0 <=i < 3:
        a.append(gsl_vector_get(v, i))
    gsl_vector_free(v)
    return a

def t_gsl_vector_memcpy():
    cdef gsl_vector *v, *w
    v = gsl_vector_alloc(3)
    w = gsl_vector_alloc(3)
    cdef int i
    for i from 0 <=i < 3:
        gsl_vector_set(v, i, i + 0.1)
    gsl_vector_memcpy(w, v)
    a = []
    for i from 0 <=i < 3:
        a.append(gsl_vector_get(w, i) - gsl_vector_get(v, i))
    gsl_vector_free(v)
    gsl_vector_free(w)
    return a

def t_gsl_vector_reverse():
    cdef gsl_vector *v
    v = gsl_vector_alloc(2)
    cdef int i
    for i from 0 <=i < 2:
        gsl_vector_set(v, i, i + 0.5)
    gsl_vector_reverse(v)
    a = (gsl_vector_get(v, 0) - 1.5, gsl_vector_get(v, 1) - 0.5)
    gsl_vector_free(v)
    return a

def t_gsl_vector_swap():
    cdef gsl_vector *v, *w
    v = gsl_vector_alloc(3)
    w = gsl_vector_calloc(3)
    cdef int i
    for i from 0 <=i < 3:
        gsl_vector_set(v, i, i + 0.1)
    gsl_vector_swap(w, v)
    a = []
    for i from 0 <=i < 3:
        a.append(gsl_vector_get(w, i) - i - 0.1)
    gsl_vector_free(v)
    gsl_vector_free(w)
    return a

def t_gsl_vector_swap_elements():
    cdef gsl_vector *v
    v = gsl_vector_alloc(3)
    cdef int i
    for i from 0 <=i < 3:
        gsl_vector_set(v, i, i + 0.1)
    gsl_vector_swap_elements(v,0,1)
    a = (gsl_vector_get(v,0) - 1.1,
      gsl_vector_get(v,1) - 0.1, gsl_vector_get(v,2) -2.1)
    gsl_vector_free(v)
    return a

def t_gsl_vector_max():
    cdef gsl_vector *v
    v = gsl_vector_alloc(3)
    cdef int i
    a = [3, 4.1,1.2]
    for i from 0 <=i < 3:
        gsl_vector_set(v,i, a[i])
    cdef double min_out, max_out
    gsl_vector_minmax(v, &min_out, &max_out)
    ary = (gsl_vector_max(v) - 4.1, gsl_vector_min(v) - 1.2,
            min_out - 1.2, max_out - 4.1)
    gsl_vector_free(v)
    return ary

def t_gsl_vector_max_index():
    cdef gsl_vector *v
    v = gsl_vector_alloc(3)
    cdef int i
    a = [3, 4.1,1.2]
    for i from 0 <=i < 3:
        gsl_vector_set(v,i, a[i])
    cdef size_t imin, imax
    gsl_vector_minmax_index(v, &imin, &imax)
    ary = (gsl_vector_max_index(v) - 1,
            gsl_vector_min_index(v) - 2, imin - 2, imax - 1)
    gsl_vector_free(v)
    return ary


def t_gsl_vector_add():
    a = [3, 4.1,1.2]
    b = [1, 2, 3.3]
    c = [4, 6.1, 4.5]
    cdef gsl_vector *v1, *v2
    v1 = gsl_vector_alloc(3)
    v2 = gsl_vector_alloc(3)
    cdef int i
    for i from 0 <=i < 3:
        gsl_vector_set(v1,i, a[i])
        gsl_vector_set(v2,i, b[i])
    gsl_vector_add(v1,v2)
    ary = []
    for i from 0 <=i < 3:
        ary.append(gsl_vector_get(v1,i) - c[i])
    gsl_vector_free(v1)
    gsl_vector_free(v2)
    return ary


def t_gsl_vector_sub():
    a = [3, 4.1,1.2]
    b = [1, 2, 3.3]
    c = [4, 6.1, 4.5]
    cdef gsl_vector *v1, *v2
    v1 = gsl_vector_alloc(3)
    v2 = gsl_vector_alloc(3)
    cdef int i
    for i from 0 <=i < 3:
        gsl_vector_set(v1,i, c[i])
        gsl_vector_set(v2,i, b[i])
    gsl_vector_sub(v1,v2)
    ary = []
    for i from 0 <=i < 3:
        ary.append(gsl_vector_get(v1,i) - a[i])
    gsl_vector_free(v1)
    gsl_vector_free(v2)
    return ary

def t_gsl_vector_mul():
    a = [3, 4.1,1.2]
    b = [1, 2, 3.3]
    c = [3, 8.2, 3.96]
    cdef gsl_vector *v1, *v2
    v1 = gsl_vector_alloc(3)
    v2 = gsl_vector_alloc(3)
    cdef int i
    for i from 0 <=i < 3:
        gsl_vector_set(v1,i, a[i])
        gsl_vector_set(v2,i, b[i])
    gsl_vector_mul(v1,v2)
    ary = []
    for i from 0 <=i < 3:
        ary.append(gsl_vector_get(v1,i) - c[i])
    gsl_vector_free(v1)
    gsl_vector_free(v2)
    return ary

def t_gsl_vector_div():
    a = [3, 4.1,1.2]
    b = [1, 2, 3.3]
    c = [3, 8.2, 3.96]
    cdef gsl_vector *v1, *v2
    v1 = gsl_vector_alloc(3)
    v2 = gsl_vector_alloc(3)
    cdef int i
    for i from 0 <=i < 3:
        gsl_vector_set(v1,i, c[i])
        gsl_vector_set(v2,i, b[i])
    gsl_vector_div(v1,v2)
    ary = []
    for i from 0 <=i < 3:
        ary.append(gsl_vector_get(v1,i) - a[i])
    gsl_vector_free(v1)
    gsl_vector_free(v2)
    return ary

def t_gsl_vector_scale():
    a = [3, 4.1,1.2]
    b = [3.3, 4.51, 1.32]
    cdef gsl_vector *v1
    v1 = gsl_vector_alloc(3)
    cdef int i
    for i from 0 <=i < 3:
        gsl_vector_set(v1,i, a[i])
    gsl_vector_scale(v1,1.1)
    ary = []
    for i from 0 <=i < 3:
        ary.append(gsl_vector_get(v1,i) - b[i])
    gsl_vector_free(v1)
    return ary

def t_gsl_vector_add_constant():
    a = [3, 4.1,1.2]
    b = [3.3, 4.4, 1.5]
    cdef gsl_vector *v1
    v1 = gsl_vector_alloc(3)
    cdef int i
    for i from 0 <=i < 3:
        gsl_vector_set(v1,i, a[i])
    gsl_vector_add_constant(v1,0.3)
    ary = []
    for i from 0 <=i < 3:
        ary.append(gsl_vector_get(v1,i) - b[i])
    gsl_vector_free(v1)
    return ary

def t_gsl_vector_isnull():
    cdef gsl_vector *v1, *v2
    v1 = gsl_vector_calloc(3)
    v2 = gsl_vector_calloc(3)
    gsl_vector_set(v1, 1, 0.01)
    a = (gsl_vector_isnull(v1), gsl_vector_isnull(v2) - 1)
    gsl_vector_free(v1)
    gsl_vector_free(v2)
    return a

def t_gsl_vector_subvector():
    cdef gsl_vector *v1, *v2
    cdef gsl_vector_view vw
    v1 = gsl_vector_alloc(10)
    cdef int i
    for i from 0 <=i < 10:
        gsl_vector_set(v1, i, i + 0.1)
    vw = gsl_vector_subvector(v1, 4, 3)
    v2 = &vw.vector
    a = []
    for i from 0 <=i < 3:
        a.append(gsl_vector_get(v2, i) - i - 4 - 0.1)
    gsl_vector_free(v1)
    return a

def t_gsl_vector_subvector_with_stride1():
    cdef gsl_vector *v1, *v2
    cdef gsl_vector_view vw
    v1 = gsl_vector_alloc(10)
    cdef int i
    for i from 0 <=i < 10:
        gsl_vector_set(v1, i, i + 0.1)
    vw = gsl_vector_subvector_with_stride(v1, 4, 2, 3)
    v2 = &vw.vector
    a = []
    for i from 0 <=i < 3:
        a.append(gsl_vector_get(v2, i) - 2*i - 4 - 0.1)
    gsl_vector_free(v1)
    return a

def t_gsl_vector_subvector_with_stride2():
    cdef gsl_vector *v1, *v2
    cdef gsl_vector_view vw
    v1 = gsl_vector_alloc(10)
    cdef int i
    for i from 0 <=i < 10:
        gsl_vector_set(v1, i, i + 0.1)
    cdef size_t offset
    offset = 0
    vw = gsl_vector_subvector_with_stride(v1, offset, 2, 5)
    v2 = &vw.vector
    a = []
    for i from 0 <=i < 3:
        a.append(gsl_vector_get(v2, i) - 2*i - offset - 0.1)
    gsl_vector_free(v1)
    return a

def t_gsl_vector_view_array():
    cdef gsl_vector_view vw
    cdef gsl_vector *v1
    cdef double b[3]
    b[0] = 1.1; b[1] = 2.1; b[2] = 3.1
    vw = gsl_vector_view_array(b, 3)
    v1 = &vw.vector
    a = []
    for i from 0 <=i < 3:
        a.append(gsl_vector_get(v1, i) - i - 1.1)
    return a


def t_gsl_vector_view_array_with_stride():
    cdef gsl_vector_view vw
    cdef gsl_vector *v1
    cdef double b[3]
    b[0] = 1.1; b[1] = 2.1; b[2] = 3.1; b[3] = 4.1
    vw = gsl_vector_view_array_with_stride(b, 2, 3)
    v1 = &vw.vector
    a = []
    for i from 0 <=i < 2:
        a.append(gsl_vector_get(v1, i) - 2*i - 1.1)
    return a
