from cython_gsl cimport *

def t_gsl_matrix_alloc():
    cdef gsl_matrix * m
    m = gsl_matrix_alloc (10, 3)
    cdef int i,j
    for i from 0 <= i < 10:
        for j from 0 <= j < 3:
            gsl_matrix_set (m, i, j, 0.23 + 100*i + j)
    a = []
    for i from 0 <= i < 10:
        for j from 0 <= j < 3:
            a.append(gsl_matrix_get (m, i, j) - 0.23 - 100*i - j)
    gsl_matrix_free(m)
    return a

def t_gsl_matrix_calloc():
    cdef gsl_matrix * m
    m = gsl_matrix_calloc (3, 3)
    cdef int i,j
    a = []
    for i from 0 <= i < 3:
        for j from 0 <= j < 3:
            a.append(gsl_matrix_get(m, i, j))
    gsl_matrix_free(m)
    return a

def t_gsl_matrix_max():
    cdef gsl_matrix * m
    m = gsl_matrix_alloc (2, 3)
    cdef double mmin, mmax
    cdef size_t i,j
    cdef size_t i1, j1, i2, j2
    for i from 0 <= i < 2:
        for j from 0 <= j < 3:
            gsl_matrix_set (m, i, j, 0.2 + i + j)
    ary = [gsl_matrix_min(m) - 0.2, gsl_matrix_max(m) - 3.2]
    gsl_matrix_minmax(m, &mmin, &mmax)
    ary.extend([mmin - 0.2, mmax - 3.2])
    gsl_matrix_max_index(m, &i1, &j1)
    ary.extend([i1 - 1, j1 - 2])
    gsl_matrix_min_index(m, &i1, &j1)
    ary.extend([i1, j1])
    gsl_matrix_minmax_index(m, &i1, &j1, &i2, &j2)
    ary.extend([i1, j1, i2 - 1, j2 - 2])
    gsl_matrix_free(m)
    return ary

def t_gsl_matrix_isnull():
    cdef gsl_matrix * m
    m = gsl_matrix_calloc (2, 3)
    ary = [gsl_matrix_isnull(m) - 1, gsl_matrix_get(m, 0,0)]
    gsl_matrix_free(m)
    return ary


def t_gsl_matrix_add():
    cdef gsl_matrix * m1, * m2
    m1 = gsl_matrix_alloc (2, 3)
    m2 = gsl_matrix_alloc (2, 3)
    cdef int i,j
    a = [[0,0,0],[0,0,0]]
    for i from 0 <= i < 2:
        for j from 0 <= j < 3:
            gsl_matrix_set (m1, i, j, 0.2 + i + j)
            gsl_matrix_set (m2, i, j, 0.3 + i + j)
            a[i][j] = 2*i + 2*j + 0.5
    ary = []
    gsl_matrix_add(m1, m2)
    for i from 0 <= i < 2:
        for j from 0 <= j < 3:
            ary.append(gsl_matrix_get(m1, i, j) - a[i][j])
            ary.append(gsl_matrix_get(m2, i, j) - 0.3 - i -j)
    gsl_matrix_sub(m1, m2)
    for i from 0 <= i < 2:
        for j from 0 <= j < 3:
            ary.append(gsl_matrix_get(m1, i, j) - 0.2 - i -j)
            ary.append(gsl_matrix_get(m2, i, j) - 0.3 - i -j)
    gsl_matrix_free(m1)
    gsl_matrix_free(m2)
    return ary

def t_gsl_matrix_mul_elements():
    cdef gsl_matrix * m1, * m2
    m1 = gsl_matrix_alloc (2, 3)
    m2 = gsl_matrix_alloc (2, 3)
    cdef int i,j
    a = [[0,0,0],[0,0,0]]
    for i from 0 <= i < 2:
        for j from 0 <= j < 3:
            gsl_matrix_set (m1, i, j, 0.2 + i + j)
            gsl_matrix_set (m2, i, j, 0.3 + i + j)
            a[i][j] = (i + j + 0.2)*(0.3 + i + j)
    ary = []
    gsl_matrix_mul_elements(m1, m2)
    for i from 0 <= i < 2:
        for j from 0 <= j < 3:
            ary.append(gsl_matrix_get(m1, i, j) - a[i][j])
            ary.append(gsl_matrix_get(m2, i, j) - 0.3 - i -j)
    gsl_matrix_div_elements(m1, m2)
    for i from 0 <= i < 2:
        for j from 0 <= j < 3:
            ary.append(gsl_matrix_get(m1, i, j) - 0.2 - i -j)
            ary.append(gsl_matrix_get(m2, i, j) - 0.3 - i -j)
    gsl_matrix_free(m1)
    gsl_matrix_free(m2)
    return ary

def t_gsl_matrix_scale():
    cdef gsl_matrix * m1
    m1 = gsl_matrix_alloc (2, 3)
    cdef int i,j
    cdef double d1
    d1 = 0.251
    a = [[0,0,0],[0,0,0]]
    for i from 0 <= i < 2:
        for j from 0 <= j < 3:
            gsl_matrix_set (m1, i, j, 0.2 + i + j)
            a[i][j] = (i + j + 0.2)*d1
    ary = []
    gsl_matrix_scale(m1, d1)
    for i from 0 <= i < 2:
        for j from 0 <= j < 3:
            ary.append(gsl_matrix_get(m1, i, j) - a[i][j])
    gsl_matrix_free(m1)
    return ary

def t_gsl_matrix_add_constant():
    cdef gsl_matrix * m1
    m1 = gsl_matrix_alloc (2, 3)
    cdef int i,j
    cdef double d1
    d1 = 0.251
    a = [[0,0,0],[0,0,0]]
    for i from 0 <= i < 2:
        for j from 0 <= j < 3:
            gsl_matrix_set (m1, i, j, 0.2 + i + j)
            a[i][j] = (i + j + 0.2) + d1
    ary = []
    gsl_matrix_add_constant(m1, d1)
    for i from 0 <= i < 2:
        for j from 0 <= j < 3:
            ary.append(gsl_matrix_get(m1, i, j) - a[i][j])
    gsl_matrix_free(m1)
    return ary

def t_gsl_matrix_add_diagonal():
    cdef gsl_matrix * m1
    m1 = gsl_matrix_alloc (2, 3)
    cdef int i,j
    cdef double d1
    d1 = 0.251
    a = [[0,0,0],[0,0,0]]
    for i from 0 <= i < 2:
        for j from 0 <= j < 3:
            gsl_matrix_set (m1, i, j, 0.2 + i + j)
            a[i][j] = (i + j + 0.2)
    for i from 0 <= i < 2:
        a[i][i] = a[i][i] + d1
    ary = []
    gsl_matrix_add_diagonal(m1, d1)
    for i from 0 <= i < 2:
        for j from 0 <= j < 3:
            ary.append(gsl_matrix_get(m1, i, j) - a[i][j])
    gsl_matrix_free(m1)
    return ary

def t_gsl_matrix_memcpy():
    cdef gsl_matrix * m1, * m2
    m1 = gsl_matrix_alloc (2, 3)
    m2 = gsl_matrix_alloc (2, 3)
    cdef int i,j
    for i from 0 <= i < 2:
        for j from 0 <= j < 3:
            gsl_matrix_set (m1, i, j, 0.2 + i + j)
    ary = []
    gsl_matrix_memcpy(m2, m1)
    for i from 0 <= i < 2:
        for j from 0 <= j < 3:
            ary.append(gsl_matrix_get(m1, i, j) - gsl_matrix_get(m2, i, j))
    gsl_matrix_free(m1)
    gsl_matrix_free(m2)
    return ary

def t_gsl_matrix_swap():
    cdef gsl_matrix * m1, * m2
    m1 = gsl_matrix_alloc (2, 3)
    m2 = gsl_matrix_alloc (2, 3)
    cdef int i,j
    for i from 0 <= i < 2:
        for j from 0 <= j < 3:
            gsl_matrix_set (m1, i, j, 0.2 + i + j)
            gsl_matrix_set (m2, i, j, 0.3 + i + j)
    ary = []
    gsl_matrix_swap(m1, m2)
    for i from 0 <= i < 2:
        for j from 0 <= j < 3:
            ary.append(gsl_matrix_get(m1, i, j) - 0.3 - i -j)
            ary.append(gsl_matrix_get(m2, i, j) - 0.2 - i -j)
    gsl_matrix_free(m1)
    gsl_matrix_free(m2)
    return ary

def t_gsl_matrix_swap_rows():
    cdef gsl_matrix * m1
    m1 = gsl_matrix_alloc (4, 3)
    cdef int i,j
    cdef double d1
    for i from 0 <= i < 4:
        for j from 0 <= j < 3:
            gsl_matrix_set (m1, i, j, 0.2 + i + j)
    ary = []
    gsl_matrix_swap_rows(m1, 1, 3)
    a = [0,3,2,1]
    for i from 0 <= i < 4:
        for j from 0 <= j < 3:
            ary.append(gsl_matrix_get(m1, i, j) - 0.2 - a[i] -j)
    gsl_matrix_free(m1)
    return ary

def t_gsl_matrix_swap_columns():
    cdef gsl_matrix * m1
    m1 = gsl_matrix_alloc (4, 5)
    cdef int i,j
    cdef double d1
    for i from 0 <= i < 4:
        for j from 0 <= j < 5:
            gsl_matrix_set (m1, i, j, 0.2 + i + j)
    ary = []
    gsl_matrix_swap_columns(m1, 1, 3)
    a = [0,3,2,1, 4]
    for i from 0 <= i < 4:
        for j from 0 <= j < 5:
            ary.append(gsl_matrix_get(m1, i, j) - 0.2 - i - a[j])
    gsl_matrix_free(m1)
    return ary

def t_gsl_matrix_set_identity():
    cdef gsl_matrix * m1,  * m2
    m1 = gsl_matrix_calloc (3, 3)
    m2 = gsl_matrix_alloc (3, 3)
    cdef int i,j
    gsl_matrix_set_identity(m2)
    for i from 0 <= i < 3:
        gsl_matrix_set(m1, i, i, 1.0)
    ary = []
    gsl_matrix_sub(m1, m2)
    for i from 0 <= i < 3:
        for j from 0 <= j < 3:
            ary.append(gsl_matrix_get(m1, i, j))
    gsl_matrix_set_all(m1, 1.3)
    for i from 0 <= i < 3:
        for j from 0 <= j < 3:
            ary.append(gsl_matrix_get(m1, i, j) - 1.3)
    gsl_matrix_set_zero(m1)
    for i from 0 <= i < 3:
        for j from 0 <= j < 3:
            ary.append(gsl_matrix_get(m1, i, j))
    gsl_matrix_free(m1)
    gsl_matrix_free(m2)
    return ary

def t_gsl_matrix_transpose():
    cdef gsl_matrix * m
    m = gsl_matrix_alloc (3, 3)
    cdef int i,j
    for i from 0 <= i < 3:
        for j from 0 <= j < 3:
            gsl_matrix_set(m, i, j, i + 2*j + 0.1)
    ary = []
    gsl_matrix_transpose(m)
    for i from 0 <= i < 3:
        for j from 0 <= j < 3:
            ary.append(gsl_matrix_get(m, i, j) - 2*i - j - 0.1)
    gsl_matrix_free(m)
    return ary

def t_gsl_matrix_transpose_memcpy():
    cdef gsl_matrix * m1, * m2
    m1 = gsl_matrix_alloc (3, 4)
    m2 = gsl_matrix_alloc (4, 3)
    cdef int i,j
    for i from 0 <= i < 3:
        for j from 0 <= j < 4:
            gsl_matrix_set(m1, i, j, i + 2*j + 0.1)
    ary = []
    gsl_matrix_transpose_memcpy(m2, m1)
    for i from 0 <= i < 4:
        for j from 0 <= j < 3:
            ary.append(gsl_matrix_get(m2, i, j) - 2*i - j - 0.1)
            ary.append(gsl_matrix_get(m1, j, i) - 2*i - j - 0.1)
    gsl_matrix_free(m1)
    gsl_matrix_free(m2)
    return ary

def t_gsl_matrix_row():
    cdef gsl_matrix * m1
    cdef gsl_vector * v1
    cdef gsl_vector_view vw1
    m1 = gsl_matrix_alloc (3, 4)
    cdef int i,j
    for i from 0 <= i < 3:
        for j from 0 <= j < 4:
            gsl_matrix_set(m1, i, j, i + 2*j + 0.1)
    vw1 = gsl_matrix_row(m1, 1)
    v1 = &vw1.vector
    ary = []
    for j from 0 <= j < 4:
        ary.append(gsl_vector_get(v1, j) - 1.1 - 2*j)
    gsl_matrix_free(m1)
    return ary

def t_gsl_matrix_column():
    cdef gsl_matrix * m1
    cdef gsl_vector * v1
    cdef gsl_vector_view vw1
    m1 = gsl_matrix_alloc (3, 4)
    cdef int i,j
    for i from 0 <= i < 3:
        for j from 0 <= j < 4:
            gsl_matrix_set(m1, i, j, i + 2*j + 0.1)
    vw1 = gsl_matrix_column(m1, 1)
    v1 = &vw1.vector
    ary = []
    for i from 0 <= i < 3:
        ary.append(gsl_vector_get(v1, i) - 2.1 - i)
    gsl_matrix_free(m1)
    return ary

def t_gsl_matrix_submatrix():
    cdef gsl_matrix * m1, * m2
    cdef gsl_matrix_view mw1
    m1 = gsl_matrix_alloc (3, 4)
    cdef int i,j
    for i from 0 <= i < 3:
        for j from 0 <= j < 4:
            gsl_matrix_set(m1, i, j, i + 2*j + 0.1)
    mw1 = gsl_matrix_submatrix(m1, 1, 1, 2, 2)
    m2 = &mw1.matrix
    a = [[0,0],[0,0]]
    for i from 1 <= i < 3:
        for j from 1 <= j < 3:
            a[i-1][j-1] = i + 2*j + 0.1
    ary = []
    for i from 0 <= i < 2:
        for j from 0 <= j < 2:
            ary.append(gsl_matrix_get(m2, i,j) - a[i][j])
    #
    # show that m2 shares the data of m1
    gsl_matrix_set(m2, 0,1, 123.1)
    ary.append(gsl_matrix_get(m1, 1, 2) - 123.1)
    gsl_matrix_free(m1)
    return ary

def t_gsl_matrix_diagonal():
    cdef gsl_matrix * m1
    cdef gsl_vector * v1
    cdef gsl_vector_view vw1
    m1 = gsl_matrix_alloc (3, 4)
    cdef int i,j
    for i from 0 <= i < 3:
        for j from 0 <= j < 4:
            gsl_matrix_set(m1, i, j, i + 2*j + 0.1)
    vw1 = gsl_matrix_diagonal(m1)
    v1 = &vw1.vector
    a = []
    for i from 0 <= i < 3:
        a.append(i + 2*i + 0.1)
    ary = []
    for i from 0 <= i < 3:
        ary.append(gsl_vector_get(v1, i) - a[i])
    gsl_matrix_free(m1)
    return ary

def t_gsl_matrix_subdiagonal():
    cdef gsl_matrix * m1
    cdef gsl_vector * v1
    cdef gsl_vector_view vw1
    m1 = gsl_matrix_alloc (3, 4)
    cdef int i,j
    for i from 0 <= i < 3:
        for j from 0 <= j < 4:
            gsl_matrix_set(m1, i, j, i + 2*j + 0.1)
    vw1 = gsl_matrix_subdiagonal(m1, 1)
    v1 = &vw1.vector
    a = []
    for i from 0 <= i < 3:
        a.append(i + 1 + 2*i + 0.1)
    ary = []
    for i from 0 <= i < 2:
        ary.append(gsl_vector_get(v1, i) - a[i])
    gsl_matrix_free(m1)
    return ary

def t_gsl_matrix_superdiagonal():
    cdef gsl_matrix * m1
    cdef gsl_vector * v1
    cdef gsl_vector_view vw1
    m1 = gsl_matrix_alloc (3, 4)
    cdef int i,j
    for i from 0 <= i < 3:
        for j from 0 <= j < 4:
            gsl_matrix_set(m1, i, j, i + 2*j + 0.1)
    vw1 = gsl_matrix_superdiagonal(m1, 1)
    v1 = &vw1.vector
    a = []
    for i from 0 <= i < 3:
        a.append(i + 2*(i+1) + 0.1)
    ary = []
    for i from 0 <= i < 2:
        ary.append(gsl_vector_get(v1, i) - a[i])
    gsl_matrix_free(m1)
    return ary

def t_gsl_matrix_view_array():
    cdef gsl_matrix * m
    cdef gsl_matrix_view mw
    cdef double b[4]
    b[0] = 1.1; b[1] = 2.1; b[2] = 3.1; b[3] = 4.1
    mw = gsl_matrix_view_array(b, 2, 2)
    m = &mw.matrix
    a = [[1.1, 2.1], [3.1, 4.1]]
    ary = []
    for i from 0 <= i < 2:
        for j from 0 <= j < 2:
            ary.append(gsl_matrix_get(m, i, j) - a[i][j])
    return ary

def t_gsl_matrix_view_vector():
    cdef gsl_matrix * m
    cdef gsl_matrix_view mw
    cdef gsl_vector * v
    v = gsl_vector_alloc(4)
    for i from 0 <= i < 4:
        gsl_vector_set(v, i, i + 1.1)
    mw = gsl_matrix_view_vector(v, 2, 2)
    m = &mw.matrix
    a = [[1.1, 2.1], [3.1, 4.1]]
    ary = []
    for i from 0 <= i < 2:
        for j from 0 <= j < 2:
            ary.append(gsl_matrix_get(m, i, j) - a[i][j])
    return ary

def t_gsl_matrix_fprintf():
    cdef gsl_matrix *m
    m = gsl_matrix_alloc(3, 3)
    gsl_matrix_set_identity(m)
    cdef FILE * f
    f = fopen ("test.dat", "w")
    gsl_matrix_fprintf (f, m, "%.5g")
    fclose (f)
    cdef FILE * f2
    f2 = fopen ("test.dat", "r")
    cdef gsl_matrix *m2
    m2 = gsl_matrix_alloc(3, 3)
    gsl_matrix_fscanf(f2, m2)
    fclose (f2)
    ary = []
    for i from 0 <= i < 3:
        for j from 0 <= j < 3:
            ary.append(gsl_matrix_get(m,i,j) - gsl_matrix_get(m2,i,j))
    gsl_matrix_free(m)
    gsl_matrix_free(m2)
    return ary

def t_gsl_matrix_set_row():
    cdef gsl_matrix * m
    m = gsl_matrix_alloc (3, 3)
    cdef int i,j
    for i from 0 <= i < 3:
        for j from 0 <= j < 3:
            gsl_matrix_set (m, i, j, 0.23 + 100*i + j)
    cdef gsl_vector * v
    v = gsl_vector_alloc(3)
    for i from 0 <= i < 3:
        gsl_vector_set(v,i, 2*i)
    gsl_matrix_set_row(m, 1, v)
    ary = []
    for i from 0 <= i < 3:
        for j from 0 <= j < 3:
            if i != 1:
                ary.append(gsl_matrix_get(m,i,j) - 0.23 - 100*i - j)
            if i == 1:
                ary.append(gsl_matrix_get(m,i,j) - 2*j)
    gsl_matrix_free(m)
    gsl_vector_free(v)
    return ary
