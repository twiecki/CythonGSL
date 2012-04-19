from cython_gsl cimport *

def t_gsl_block_complex():
    cdef gsl_block_complex * b
    b = gsl_block_complex_alloc(100)
    cdef size_t size1, size2
    size1 = b.size
    size2 = gsl_block_complex_size(b)
    return (size1 - 100, size2 - 100)


def t_gsl_vector_complex_set():
    cdef gsl_vector_complex *v
    cdef gsl_complex z
    v = gsl_vector_complex_alloc(3)
    cdef int i
    for i from 0 <=i < 3:
        gsl_vector_complex_set(v, i, gsl_complex_rect(i + 0.1, i - 0.1))
    a = []
    for i from 0 <=i < 3:
        z = gsl_vector_complex_get(v, i)
        a.append(GSL_REAL(z) - i - 0.1)
        a.append(GSL_IMAG(z) - i + 0.1)
    gsl_vector_complex_free(v)
    return a

def t_gsl_vector_complex_fprintf():
    cdef int i
    cdef gsl_vector_complex *v
    v = gsl_vector_complex_alloc(100)

    for i from 0 <= i < 100:
        gsl_vector_complex_set (v, i, gsl_complex_rect(1.23 + i,i))

    cdef FILE * f
    f = fopen ("test.dat", "w")
    gsl_vector_complex_fprintf (f, v, "%.5g")
    fclose (f)

    cdef gsl_vector_complex *v2
    v2 = gsl_vector_complex_alloc(10)


    cdef FILE * f2
    f2 = fopen ("test.dat", "r")
    gsl_vector_complex_fscanf (f2, v)
    fclose (f2)
    cdef gsl_complex z
    s1 = ""
    for i from 0 <= i < 10:
        z = gsl_vector_complex_get(v, i)
        s1 =  s1 + "%.5g  %.5g\n"  %(GSL_REAL(z), GSL_IMAG(z))
    return s1 == '1.23  0\n2.23  1\n3.23  2\n4.23  3\n5.23  4\n6.23  5\n7.23  6\n8.23  7\n9.23  8\n10.23  9\n'

def t_gsl_vector_complex_set_zero():
    cdef gsl_vector_complex *v
    cdef gsl_complex z
    v = gsl_vector_complex_alloc(3)
    cdef int i
    for i from 0 <=i < 3:
        gsl_vector_complex_set(v, i, gsl_complex_rect(1.23 + i,i))
    gsl_vector_complex_set_zero(v)
    a = []
    for i from 0 <=i < 3:
        z = gsl_vector_complex_get(v, i)
        a.append(GSL_REAL(z))
        a.append(GSL_IMAG(z))
    return a

def t_gsl_vector_complex_set_all():
    cdef gsl_vector_complex *v
    cdef gsl_complex z
    v = gsl_vector_complex_alloc(3)
    gsl_vector_complex_set_all(v, gsl_complex_rect(1.23, 4.5))
    a = []
    for i from 0 <=i < 3:
        z = gsl_vector_complex_get(v, i)
        a.append(GSL_REAL(z) - 1.23)
        a.append(GSL_IMAG(z) - 4.5)
    return a

def t_gsl_vector_complex_set_basis():
    cdef gsl_vector_complex *v
    cdef gsl_complex z
    v = gsl_vector_complex_calloc(3)
    gsl_vector_complex_set_basis(v, 1)
    a = []
    z = gsl_vector_complex_get(v,0)
    a.append(GSL_REAL(z))
    a.append(GSL_IMAG(z))
    z = gsl_vector_complex_get(v,1)
    a.append(GSL_REAL(z) - 1)
    a.append(GSL_IMAG(z))
    z = gsl_vector_complex_get(v,2)
    a.append(GSL_REAL(z))
    a.append(GSL_IMAG(z))
    return a

def t_gsl_vector_complex_memcpy():
    cdef gsl_vector_complex *v1, *v2
    cdef gsl_complex z
    v1 = gsl_vector_complex_calloc(3)
    v2 = gsl_vector_complex_calloc(3)
    gsl_vector_complex_set_basis(v1, 1)
    gsl_vector_complex_memcpy(v2, v1)
    a = []
    z = gsl_vector_complex_get(v2,0)
    a.append(GSL_REAL(z))
    a.append(GSL_IMAG(z))
    z = gsl_vector_complex_get(v2,1)
    a.append(GSL_REAL(z) - 1)
    a.append(GSL_IMAG(z))
    z = gsl_vector_complex_get(v2,2)
    a.append(GSL_REAL(z))
    a.append(GSL_IMAG(z))
    return a


def t_gsl_vector_complex_reverse():
    cdef gsl_vector_complex *v1
    cdef gsl_complex z
    v1 = gsl_vector_complex_calloc(3)
    gsl_vector_complex_set_basis(v1, 2)
    gsl_vector_complex_reverse(v1)
    a = []
    z = gsl_vector_complex_get(v1,0)
    a.append(GSL_REAL(z) - 1)
    a.append(GSL_IMAG(z))
    z = gsl_vector_complex_get(v1,1)
    a.append(GSL_REAL(z))
    a.append(GSL_IMAG(z))
    z = gsl_vector_complex_get(v1,2)
    a.append(GSL_REAL(z))
    a.append(GSL_IMAG(z))
    return a

def t_gsl_vector_complex_swap():
    cdef gsl_vector_complex *v1, *v2
    cdef gsl_complex z
    v1 = gsl_vector_complex_calloc(3)
    v2 = gsl_vector_complex_calloc(3)
    gsl_vector_complex_set_basis(v2, 1)
    gsl_vector_complex_swap(v2, v1)
    a = []
    z = gsl_vector_complex_get(v1,0)
    a.append(GSL_REAL(z))
    a.append(GSL_IMAG(z))
    z = gsl_vector_complex_get(v1,1)
    a.append(GSL_REAL(z) - 1)
    a.append(GSL_IMAG(z))
    z = gsl_vector_complex_get(v1,2)
    a.append(GSL_REAL(z))
    a.append(GSL_IMAG(z))
    z = gsl_vector_complex_get(v2,1)
    a.append(GSL_REAL(z))
    a.append(GSL_IMAG(z))
    return a


def t_gsl_vector_complex_swap_elements():
    cdef gsl_vector_complex *v1
    cdef gsl_complex z
    v1 = gsl_vector_complex_calloc(3)
    gsl_vector_complex_set_basis(v1, 1)
    gsl_vector_complex_swap_elements(v1, 1, 2)
    a = []
    z = gsl_vector_complex_get(v1,0)
    a.append(GSL_REAL(z))
    a.append(GSL_IMAG(z))
    z = gsl_vector_complex_get(v1,1)
    a.append(GSL_REAL(z))
    a.append(GSL_IMAG(z))
    z = gsl_vector_complex_get(v1,2)
    a.append(GSL_REAL(z) - 1)
    a.append(GSL_IMAG(z))
    return a

def t_gsl_vector_complex_real():
    cdef gsl_vector_complex *v1
    cdef gsl_vector *vr1, *vr2
    cdef gsl_vector_view vw1, vw2
    v1 = gsl_vector_complex_calloc(3)
    gsl_vector_complex_set(v1, 1, gsl_complex_rect(0.1, 0.2))
    vw1 = gsl_vector_complex_real(v1)
    vw2 = gsl_vector_complex_imag(v1)
    vr1 = &vw1.vector
    vr2 = &vw2.vector
    a = [0, 0.1, 0]
    b = [0, 0.2, 0]
    cdef int i
    for i from 0 <= i < 3:
        a[i] = a[i] - gsl_vector_get(vr1, i)
    for i from 0 <= i < 3:
        b[i] = b[i] - gsl_vector_get(vr2,i)
    a = a + b
    return a

def t_gsl_vector_complex_isnull():
    cdef gsl_vector_complex *v1, *v2
    v1 = gsl_vector_complex_calloc(3)
    v2 = gsl_vector_complex_calloc(3)
    gsl_vector_complex_set(v1, 1, gsl_complex_rect(0.1, 0.2))
    return (gsl_vector_complex_isnull(v1), gsl_vector_complex_isnull(v2) - 1)

def t_gsl_vector_complex_subvector():
    cdef gsl_vector_complex *v1, *v2
    cdef gsl_vector_complex_view vw
    v1 = gsl_vector_complex_alloc(10)
    v2 = gsl_vector_complex_alloc(3)
    cdef int i
    for i from 0 <=i < 10:
        gsl_vector_complex_set(v1, i, gsl_complex_rect(i + 0.1, i + 0.2))
    vw = gsl_vector_complex_subvector(v1, 4, 3)
    v2 = &vw.vector
    a = []
    for i from 0 <=i < 3:
        a.append(GSL_REAL(gsl_vector_complex_get(v2, i)) - i - 4 - 0.1)
    for i from 0 <=i < 3:
        a.append(GSL_IMAG(gsl_vector_complex_get(v2, i)) - i - 4 - 0.2)
    return a

def t_gsl_vector_complex_subvector_with_stride():
    cdef gsl_vector_complex *v1, *v2
    cdef gsl_vector_complex_view vw
    v1 = gsl_vector_complex_alloc(10)
    v2 = gsl_vector_complex_alloc(3)
    cdef int i
    for i from 0 <=i < 10:
        gsl_vector_complex_set(v1, i, gsl_complex_rect(i + 0.1, i + 0.2))
    vw = gsl_vector_complex_subvector_with_stride(v1, 4, 2, 3)
    v2 = &vw.vector
    a = []
    for i from 0 <=i < 3:
        a.append(GSL_REAL(gsl_vector_complex_get(v2, i)) - 2*i - 4 - 0.1)
    for i from 0 <=i < 3:
        a.append(GSL_IMAG(gsl_vector_complex_get(v2, i)) - 2*i - 4 - 0.2)
    return a

def t_gsl_vector_complex_view_array():
    cdef gsl_vector_complex_view vw
    cdef gsl_vector_complex *v1
    v1 = gsl_vector_complex_alloc(2)
    cdef double b[4]
    b[0] = 1.1; b[1] = 2.1; b[2] = 3.1; b[3] = 4.1
    vw = gsl_vector_complex_view_array(b, 3)
    # (b[0] + b[1] I, b[2] + b[3] I)
    v1 = &vw.vector
    a = []
    for i from 0 <=i < 2:
        a.append(GSL_REAL(gsl_vector_complex_get(v1, i)))
        a.append(GSL_IMAG(gsl_vector_complex_get(v1, i)))
    for i from 0 <=i < 4:
        a[i] =  a[i] - b[i]
    return a
