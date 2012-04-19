from cython_gsl cimport *


def t_gsl_matrix_complex_set():
    cdef gsl_matrix_complex *m
    cdef gsl_complex z
    m = gsl_matrix_complex_alloc(3,4)
    for i from 0 <= i < 3:
        for j from 0 <= j < 4:
            gsl_matrix_complex_set(m, i, j, gsl_complex_rect(i+0.1, j-0.1))
    ary = []
    for i from 0 <= i < 3:
        for j from 0 <= j < 4:
            z = gsl_complex_sub(gsl_matrix_complex_get(m,i,j), gsl_complex_rect(i+0.1, j- 0.1))
            ary.extend([GSL_REAL(z), GSL_IMAG(z)])
    gsl_matrix_complex_free(m)
    return ary

def t_gsl_matrix_complex_calloc():
    cdef gsl_matrix_complex *m
    cdef gsl_complex z
    m = gsl_matrix_complex_calloc(3,4)
    ary = []
    for i from 0 <= i < 3:
        for j from 0 <= j < 4:
            z= gsl_matrix_complex_get(m, i, j)
            ary.extend([GSL_REAL(z), GSL_IMAG(z)])
    gsl_matrix_complex_free(m)
    return ary

def t_gsl_matrix_complex_isnull():
    cdef gsl_matrix_complex *m
    cdef gsl_complex z
    m = gsl_matrix_complex_calloc(3,4)
    ary = [gsl_matrix_complex_isnull(m) - 1, GSL_REAL(gsl_matrix_complex_get(m, 0,0))]
    gsl_matrix_complex_free(m)
    return ary


def t_gsl_matrix_complex_add():
    cdef gsl_matrix_complex *m1, *m2
    cdef gsl_complex z
    m1 = gsl_matrix_complex_calloc(2,3)
    m2 = gsl_matrix_complex_calloc(2,3)
    for i from 0 <= i < 2:
        for j from 0 <= j < 3:
            gsl_matrix_complex_set(m1, i, j,
              gsl_complex_rect(0.2 + i + j, 0.4 + i + j))
            gsl_matrix_complex_set(m2, i, j,
              gsl_complex_rect(0.3 + i + j, 0.5 + i + j))
    gsl_matrix_complex_add(m1, m2)
    ary = []
    for i from 0 <= i < 2:
        for j from 0 <= j < 3:
            z = gsl_matrix_complex_get(m1, i, j)
            ary.append(GSL_REAL(z) - (0.5 + 2*i + 2*j))
            ary.append(GSL_IMAG(z) - (0.9 + 2*i + 2*j))
    gsl_matrix_complex_sub(m1, m2)
    for i from 0 <= i < 2:
        for j from 0 <= j < 3:
            z = gsl_matrix_complex_get(m1, i, j)
            ary.append(GSL_REAL(z) - (0.2 + i + j))
            ary.append(GSL_IMAG(z) - (0.4 + i + j))
    gsl_matrix_complex_free(m1)
    gsl_matrix_complex_free(m2)
    return ary

def t_gsl_matrix_complex_mul_elements():
    cdef gsl_matrix_complex *m1, *m2
    cdef gsl_complex z, z1,z2,z3
    m1 = gsl_matrix_complex_calloc(2,3)
    m2 = gsl_matrix_complex_calloc(2,3)
    for i from 0 <= i < 2:
        for j from 0 <= j < 3:
            gsl_matrix_complex_set(m1, i, j,
              gsl_complex_rect(0.2 + i + j, 0.4 + i + j))
            gsl_matrix_complex_set(m2, i, j,
              gsl_complex_rect(0.3 + i + j, 0.5 + i + j))
    gsl_matrix_complex_mul_elements(m1, m2)
    ary = []
    for i from 0 <= i < 2:
        for j from 0 <= j < 3:
            z = gsl_matrix_complex_get(m1, i, j)
            z1 = gsl_complex_rect(0.2 + i + j, 0.4 + i + j)
            z2 = gsl_complex_rect(0.3 + i + j, 0.5 + i + j)
            z3 = gsl_complex_mul(z1, z2)
            ary.append(GSL_REAL(z) - GSL_REAL(z3))
            ary.append(GSL_IMAG(z) - GSL_IMAG(z3))
    gsl_matrix_complex_div_elements(m1, m2)
    for i from 0 <= i < 2:
        for j from 0 <= j < 3:
            z = gsl_matrix_complex_get(m1, i, j)
            ary.append(GSL_REAL(z) - (0.2 + i + j))
            ary.append(GSL_IMAG(z) - (0.4 + i + j))
    gsl_matrix_complex_free(m1)
    gsl_matrix_complex_free(m2)
    return ary

def t_gsl_matrix_complex_scale():
    cdef gsl_matrix_complex *m1
    cdef gsl_complex z, z1, z2
    z1 = gsl_complex_rect(0.7,0.8)
    m1 = gsl_matrix_complex_calloc(2,3)
    for i from 0 <= i < 2:
        for j from 0 <= j < 3:
            gsl_matrix_complex_set(m1, i, j,
              gsl_complex_rect(0.2 + i + j, 0.4 + i + j))
    gsl_matrix_complex_scale(m1, z1)
    ary = []
    for i from 0 <= i < 2:
        for j from 0 <= j < 3:
            z = gsl_matrix_complex_get(m1, i, j)
            z2 = gsl_complex_mul(z1,  gsl_complex_rect(0.2 + i + j, 0.4 + i + j))
            ary.append(GSL_REAL(z) - GSL_REAL(z2))
            ary.append(GSL_IMAG(z) - GSL_IMAG(z2))
    gsl_matrix_complex_free(m1)
    return ary

def t_gsl_matrix_complex_add_constant():
    cdef gsl_matrix_complex *m1
    cdef gsl_complex z, z1, z2
    z1 = gsl_complex_rect(0.7,0.8)
    m1 = gsl_matrix_complex_calloc(2,3)
    for i from 0 <= i < 2:
        for j from 0 <= j < 3:
            gsl_matrix_complex_set(m1, i, j,
              gsl_complex_rect(0.2 + i + j, 0.4 + i + j))
    gsl_matrix_complex_add_constant(m1, z1)
    ary = []
    for i from 0 <= i < 2:
        for j from 0 <= j < 3:
            z = gsl_matrix_complex_get(m1, i, j)
            z2 = gsl_complex_add(z1,  gsl_complex_rect(0.2 + i + j, 0.4 + i + j))
            ary.append(GSL_REAL(z) - GSL_REAL(z2))
            ary.append(GSL_IMAG(z) - GSL_IMAG(z2))
    gsl_matrix_complex_free(m1)
    return ary

def t_gsl_matrix_complex_add_diagonal():
    cdef gsl_matrix_complex *m1, *m2
    cdef gsl_complex z, z1, z2
    z1 = gsl_complex_rect(0.7, 0.8)
    m1 = gsl_matrix_complex_calloc(2,3)
    m2 = gsl_matrix_complex_calloc(2,3)
    for i from 0 <= i < 2:
        for j from 0 <= j < 3:
            gsl_matrix_complex_set(m1, i, j,
              gsl_complex_rect(0.2 + i + j, 0.4 + i + j))
            gsl_matrix_complex_set(m2, i, j,
              gsl_complex_rect(0.2 + i + j, 0.4 + i + j))
    # add manually in m2
    for i from 0 <= i < 2:
        gsl_matrix_complex_set(m2, i, i,
          gsl_complex_add(gsl_matrix_complex_get(m2, i, i),z1))

    gsl_matrix_complex_add_diagonal(m1, z1)
    ary = []
    for i from 0 <= i < 2:
        for j from 0 <= j < 3:
            z = gsl_matrix_complex_get(m1, i, j)
            z2 = gsl_matrix_complex_get(m2, i, j)
            ary.append(GSL_REAL(z) - GSL_REAL(z2))
            ary.append(GSL_IMAG(z) - GSL_IMAG(z2))
    gsl_matrix_complex_free(m1)
    gsl_matrix_complex_free(m2)
    return ary

def t_gsl_matrix_complex_memcpy():
    cdef gsl_matrix_complex *m1, *m2
    cdef gsl_complex z
    m1 = gsl_matrix_complex_alloc(2,3)
    m2 = gsl_matrix_complex_alloc(2,3)
    for i from 0 <= i < 2:
        for j from 0 <= j < 3:
            gsl_matrix_complex_set(m1, i, j,
              gsl_complex_rect(0.2 + i + j, 0.4 + i + j))
    gsl_matrix_complex_memcpy(m2, m1)
    ary = []
    for i from 0 <= i < 2:
        for j from 0 <= j < 3:
            z = gsl_complex_sub(gsl_matrix_complex_get(m1,i,j),
              gsl_matrix_complex_get(m2,i,j))
            ary.extend((GSL_REAL(z), GSL_IMAG(z)))
    gsl_matrix_complex_free(m1)
    gsl_matrix_complex_free(m2)
    return ary


def t_gsl_matrix_complex_swap():
    cdef gsl_matrix_complex *m1, *m2
    cdef gsl_complex z
    m1 = gsl_matrix_complex_alloc(2,3)
    m2 = gsl_matrix_complex_alloc(2,3)
    for i from 0 <= i < 2:
        for j from 0 <= j < 3:
            gsl_matrix_complex_set(m1, i, j,
              gsl_complex_rect(0.2 + i + j, 0.4 + i + j))
            gsl_matrix_complex_set(m2, i, j,
              gsl_complex_rect(0.3 + i + j, 0.5 + i + j))
    gsl_matrix_complex_swap(m1, m2)
    ary = []
    for i from 0 <= i < 2:
        for j from 0 <= j < 3:
            z = gsl_complex_sub(gsl_matrix_complex_get(m1,i,j),
              gsl_complex_rect(0.3 + i + j, 0.5 + i + j))
            ary.extend((GSL_REAL(z), GSL_IMAG(z)))
            z = gsl_complex_sub(gsl_matrix_complex_get(m2,i,j),
              gsl_complex_rect(0.2 + i + j, 0.4 + i + j))
            ary.extend((GSL_REAL(z), GSL_IMAG(z)))
    gsl_matrix_complex_free(m1)
    gsl_matrix_complex_free(m2)
    return ary

def t_gsl_matrix_complex_swap_rows():
    cdef gsl_matrix_complex *m1, *m2
    cdef gsl_complex z
    m1 = gsl_matrix_complex_alloc(4,3)
    m2 = gsl_matrix_complex_alloc(4,3)
    for i from 0 <= i < 4:
        for j from 0 <= j < 3:
            gsl_matrix_complex_set(m1, i, j,
              gsl_complex_rect(0.2 + i + j, 0.4 + i + j))
            gsl_matrix_complex_set(m2, i, j,
              gsl_complex_rect(0.2 + i + j, 0.4 + i + j))
    # swap manually rows in m2:
    for j from 0 <= j < 3:
        gsl_matrix_complex_set(m2, 1, j,
            gsl_complex_rect(0.2 + 3 + j, 0.4 + 3 + j))
        gsl_matrix_complex_set(m2, 3, j,
            gsl_complex_rect(0.2 + 1 + j, 0.4 + 1 + j))
    gsl_matrix_complex_swap_rows(m1,1, 3)
    ary = []
    for i from 0 <= i < 4:
        for j from 0 <= j < 3:
            z = gsl_complex_sub(gsl_matrix_complex_get(m1,i,j),
             gsl_matrix_complex_get(m2,i,j))
            ary.extend((GSL_REAL(z), GSL_IMAG(z)))
    gsl_matrix_complex_free(m1)
    gsl_matrix_complex_free(m2)
    return ary

def t_gsl_matrix_complex_swap_columns():
    cdef gsl_matrix_complex *m1, *m2
    cdef gsl_complex z
    m1 = gsl_matrix_complex_alloc(4,3)
    m2 = gsl_matrix_complex_alloc(4,3)
    for i from 0 <= i < 4:
        for j from 0 <= j < 3:
            gsl_matrix_complex_set(m1, i, j,
              gsl_complex_rect(0.2 + i + j, 0.4 + i + j))
            gsl_matrix_complex_set(m2, i, j,
              gsl_complex_rect(0.2 + i + j, 0.4 + i + j))
    # swap manually columss in m2:
    for i from 0 <= i < 4:
        gsl_matrix_complex_set(m2, i, 2,
          gsl_complex_rect(0.2 + i + 0, 0.4 + i + 0))
        gsl_matrix_complex_set(m2, i, 0,
          gsl_complex_rect(0.2 + i + 2, 0.4 + i + 2))
    gsl_matrix_complex_swap_columns(m1, 0, 2)
    ary = []
    for i from 0 <= i < 4:
        for j from 0 <= j < 3:
            z = gsl_complex_sub(gsl_matrix_complex_get(m1,i,j),
             gsl_matrix_complex_get(m2,i,j))
            ary.extend((GSL_REAL(z), GSL_IMAG(z)))
    gsl_matrix_complex_free(m1)
    gsl_matrix_complex_free(m2)
    return ary

def t_gsl_matrix_complex_set_identity():
    cdef gsl_matrix_complex *m1, *m2
    cdef gsl_complex z
    m1 = gsl_matrix_complex_calloc(3,3)
    m2 = gsl_matrix_complex_calloc(3,3)
    for i from 0 <= i < 3:
        gsl_matrix_complex_set(m1, i, i, gsl_complex_rect(1,0))
    gsl_matrix_complex_set_identity(m2)
    ary = []
    for i from 0 <= i < 3:
        for j from 0 <= j < 3:
            z = gsl_complex_sub(gsl_matrix_complex_get(m1,i,j),
             gsl_matrix_complex_get(m2,i,j))
            ary.extend((GSL_REAL(z), GSL_IMAG(z)))
    gsl_matrix_complex_free(m1)
    gsl_matrix_complex_free(m2)
    return ary

def t_gsl_matrix_complex_transpose():
    cdef gsl_matrix_complex *m1
    cdef gsl_complex z
    m1 = gsl_matrix_complex_alloc(3,3)
    for i from 0 <= i < 3:
        for j from 0 <= j < 3:
            gsl_matrix_complex_set(m1, i, j,
              gsl_complex_rect(0.2 + i + 2*j, 0.4 + i + 2*j))
    gsl_matrix_complex_transpose(m1)
    ary = []
    for i from 0 <= i < 3:
        for j from 0 <= j < 3:
            z = gsl_complex_sub(gsl_matrix_complex_get(m1,i,j),
              gsl_complex_rect(0.2 + 2*i + j, 0.4 + 2*i + j))
            ary.extend((GSL_REAL(z), GSL_IMAG(z)))
    gsl_matrix_complex_free(m1)
    return ary

def t_gsl_matrix_complex_transpose_memcpy():
    cdef gsl_matrix_complex *m1, *m2
    cdef gsl_complex z
    m1 = gsl_matrix_complex_alloc(3,4)
    m2 = gsl_matrix_complex_alloc(4, 3)
    for i from 0 <= i < 3:
        for j from 0 <= j < 4:
            gsl_matrix_complex_set(m1, i, j,
              gsl_complex_rect(0.2 + i + 2*j, 0.4 + i + 2*j))
    gsl_matrix_complex_transpose_memcpy(m2, m1)
    ary = []
    for i from 0 <= i < 4:
        for j from 0 <= j < 3:
            z = gsl_complex_sub(gsl_matrix_complex_get(m2,i,j),
              gsl_complex_rect(0.2 + 2*i + j, 0.4 + 2*i + j))
            ary.extend((GSL_REAL(z), GSL_IMAG(z)))
            z = gsl_complex_sub(gsl_matrix_complex_get(m1,j,i),
              gsl_complex_rect(0.2 + 2*i + j, 0.4 + 2*i + j))
            ary.extend((GSL_REAL(z), GSL_IMAG(z)))
    gsl_matrix_complex_free(m1)
    gsl_matrix_complex_free(m2)
    return ary

def t_gsl_matrix_complex_row():
    cdef gsl_matrix_complex *m1
    cdef gsl_vector_complex *v1
    cdef gsl_vector_complex_view vw1
    cdef gsl_complex z
    m1 = gsl_matrix_complex_alloc(3,4)
    for i from 0 <= i < 3:
        for j from 0 <= j < 4:
            gsl_matrix_complex_set(m1, i, j,
              gsl_complex_rect(0.2 + i + 2*j, 0.4 + i + 2*j))
    vw1 = gsl_matrix_complex_row(m1, 1)
    v1 = &vw1.vector
    ary = []
    for j from 0 <= j < 4:
        z = gsl_complex_sub(gsl_vector_complex_get(v1, j),
               gsl_matrix_complex_get(m1, 1, j))
        ary.extend((GSL_REAL(z), GSL_IMAG(z)))
    gsl_matrix_complex_free(m1)
    return ary

def t_gsl_matrix_complex_column():
    cdef gsl_matrix_complex *m1
    cdef gsl_vector_complex *v1
    cdef gsl_vector_complex_view vw1
    cdef gsl_complex z
    m1 = gsl_matrix_complex_alloc(3,4)
    for i from 0 <= i < 3:
        for j from 0 <= j < 4:
            gsl_matrix_complex_set(m1, i, j,
              gsl_complex_rect(0.2 + i + 2*j, 0.4 + i + 2*j))
    vw1 = gsl_matrix_complex_column(m1, 1)
    v1 = &vw1.vector
    ary = []
    for i from 0 <= i < 3:
        z = gsl_complex_sub(gsl_vector_complex_get(v1, i),
               gsl_matrix_complex_get(m1, i, 1))
        ary.extend((GSL_REAL(z), GSL_IMAG(z)))
    gsl_matrix_complex_free(m1)
    return ary

def t_gsl_matrix_complex_submatrix():
    cdef gsl_matrix_complex *m1, *m2
    cdef gsl_matrix_complex_view mw2
    cdef gsl_complex z
    m1 = gsl_matrix_complex_alloc(3,4)
    for i from 0 <= i < 3:
        for j from 0 <= j < 4:
            gsl_matrix_complex_set(m1, i, j,
              gsl_complex_rect(0.2 + i + 2*j, 0.4 + i + 2*j))
    mw2 = gsl_matrix_complex_submatrix(m1, 1,1,2,2)
    m2 = &mw2.matrix
    ary = []
    for i from 0 <= i < 2:
        for j from 0 <= j < 2:
            z = gsl_complex_sub(gsl_matrix_complex_get(m2, i, j),
                  gsl_matrix_complex_get(m1, i+1, j+1))
        ary.extend((GSL_REAL(z), GSL_IMAG(z)))
    gsl_matrix_complex_free(m1)
    return ary


def t_gsl_matrix_complex_diagonal():
    cdef gsl_matrix_complex *m1
    cdef gsl_vector_complex *v1
    cdef gsl_vector_complex_view vw1
    cdef gsl_complex z
    m1 = gsl_matrix_complex_alloc(3,4)
    for i from 0 <= i < 3:
        for j from 0 <= j < 4:
            gsl_matrix_complex_set(m1, i, j,
              gsl_complex_rect(0.2 + i + 2*j, 0.4 + i + 2*j))
    vw1 = gsl_matrix_complex_diagonal(m1)
    v1 = &vw1.vector
    ary = []
    for i from 0 <= i < 3:
        z = gsl_complex_sub(gsl_vector_complex_get(v1, i),
         gsl_matrix_complex_get(m1, i, i))
        ary.extend((GSL_REAL(z), GSL_IMAG(z)))
    gsl_matrix_complex_free(m1)
    return ary

def t_gsl_matrix_complex_subdiagonal():
    cdef gsl_matrix_complex *m1
    cdef gsl_vector_complex *v1
    cdef gsl_vector_complex_view vw1
    cdef gsl_complex z
    m1 = gsl_matrix_complex_alloc(3,4)
    for i from 0 <= i < 3:
        for j from 0 <= j < 4:
            gsl_matrix_complex_set(m1, i, j,
              gsl_complex_rect(0.2 + i + 2*j, 0.4 + i + 2*j))
    vw1 = gsl_matrix_complex_subdiagonal(m1, 1)
    v1 = &vw1.vector
    ary = []
    for i from 0 <= i < 2:
        z = gsl_complex_sub(gsl_vector_complex_get(v1, i),
         gsl_matrix_complex_get(m1, i + 1, i))
        ary.extend((GSL_REAL(z), GSL_IMAG(z)))
    gsl_matrix_complex_free(m1)
    return ary

def t_gsl_matrix_complex_superdiagonal():
    cdef gsl_matrix_complex *m1
    cdef gsl_vector_complex *v1
    cdef gsl_vector_complex_view vw1
    cdef gsl_complex z
    m1 = gsl_matrix_complex_alloc(3,4)
    for i from 0 <= i < 3:
        for j from 0 <= j < 4:
            gsl_matrix_complex_set(m1, i, j,
              gsl_complex_rect(0.2 + i + 2*j, 0.4 + i + 2*j))
    vw1 = gsl_matrix_complex_superdiagonal(m1, 1)
    v1 = &vw1.vector
    ary = []
    for i from 0 <= i < 3:
        z = gsl_complex_sub(gsl_vector_complex_get(v1, i),
         gsl_matrix_complex_get(m1, i, i+1))
        ary.extend((GSL_REAL(z), GSL_IMAG(z)))
    gsl_matrix_complex_free(m1)
    return ary

def t_gsl_matrix_complex_view_array():
    cdef gsl_matrix_complex *m1
    cdef double b[8]
    cdef gsl_complex z
    for i from 0 <= i < 8:
        b[i] = i + 0.1
    cdef gsl_matrix_complex_view mw1
    mw1 = gsl_matrix_complex_view_array(b, 2, 2)
    m1 = &mw1.matrix
    # m1[0,0] = b[0] + b[1] I, etc.
    ary = []
    for i from 0 <= i < 2:
        for j from 0 <= j < 2:
            z = gsl_matrix_complex_get(m1, i, j)
            ary.append(GSL_REAL(z) - b[2*2*i + 2*j])
            ary.append(GSL_IMAG(z) - b[2*2*i + 2*j + 1])
    return ary

def t_gsl_matrix_complex_view_vector():
    cdef gsl_matrix_complex *m1
    cdef gsl_vector_complex *v1
    cdef gsl_matrix_complex_view mw1
    cdef gsl_complex z
    v1 = gsl_vector_complex_alloc(4)
    for i from 0 <= i < 4:
        gsl_vector_complex_set(v1, i,
           gsl_complex_rect(2*i + 0.1, 2*i + 1.1))
    mw1 = gsl_matrix_complex_view_vector(v1, 2, 2)
    m1 = &mw1.matrix
    ary = []
    for i from 0 <= i < 2:
        for j from 0 <= j < 2:
            z = gsl_complex_sub(gsl_matrix_complex_get(m1, i, j),
              gsl_vector_complex_get(v1, 2*i + j))
            ary.extend((GSL_REAL(z), GSL_IMAG(z)))
    gsl_vector_complex_free(v1)
    return ary

def t_gsl_matrix_complex_fprintf():
    cdef gsl_matrix_complex *m1, *m2
    cdef gsl_complex z
    m1 = gsl_matrix_complex_alloc(3,3)
    m2 = gsl_matrix_complex_alloc(3,3)
    for i from 0 <= i < 3:
        for j from 0 <= j < 3:
            gsl_matrix_complex_set(m1, i, j,
              gsl_complex_rect(0.2 + i + 2*j, 0.4 + i + 2*j))
    cdef FILE * f
    f = fopen ("test.dat", "w")
    gsl_matrix_complex_fprintf(f, m1, "%.5g")
    fclose(f)
    cdef FILE * f2
    f2 = fopen ("test.dat", "r")
    gsl_matrix_complex_fscanf(f2, m2)
    fclose (f2)
    ary = []
    for i from 0 <= i < 3:
        for j from 0 <= j < 3:
            z = gsl_complex_sub(gsl_matrix_complex_get(m1, i, j),
              gsl_matrix_complex_get(m2, i, j))
    gsl_matrix_complex_free(m1)
    gsl_matrix_complex_free(m2)
    return ary
