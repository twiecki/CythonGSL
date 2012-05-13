from cython_gsl cimport *
from numpy import array

cdef gsl_vector_set_from_Num(gsl_vector *v1, Nv1):
    cdef int i
    for i from 0 <= i < Nv1.shape[0]:
        gsl_vector_set(v1, i, Nv1[i])

cdef gsl_matrix_set_from_Num(gsl_matrix * m1, Nm1):
    cdef int i,j
    for i from 0 <= i < Nm1.shape[0]:
        for j from 0 <= j < Nm1.shape[1]:
            gsl_matrix_set(m1,i, j, Nm1[i,j])

cdef gsl_matrix_complex_set_from_Num(gsl_matrix_complex * m1, Nm1):
    cdef int i,j
    cdef gsl_complex z
    for i from 0 <= i < Nm1.shape[0]:
        for j from 0 <= j < Nm1.shape[1]:
            GSL_SET_COMPLEX(&z, Nm1[i,j].real, Nm1[i,j].imag)
            gsl_matrix_complex_set(m1,i, j, z)

def t_gsl_eigen_symm():
    cdef gsl_matrix *a_diag, *a, *h, *c
    cdef gsl_vector *eval
    cdef gsl_eigen_symm_workspace *w
    a_diag = gsl_matrix_calloc(3,3)
    a = gsl_matrix_alloc(3,3)
    c = gsl_matrix_alloc(3,3)
    h = gsl_matrix_alloc(3,3)
    eval = gsl_vector_alloc(3)
    w = gsl_eigen_symm_alloc(3)
    for i in range(3):
        gsl_matrix_set(a_diag, i, i, i+1)
    Nh = array([[0.6, -0.8,0],[-0.8, -0.6,0],[0,0,1]])
    gsl_matrix_set_from_Num(h,Nh)
    # a_diag*h
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans,1.0, a_diag, h, 0.0, c)
    # h^T*a_diag*h
    gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, h, c, 0.0, a)
    gsl_eigen_symm(a, eval, w)
    ary = []
    ary.append(gsl_vector_get(eval,0) - 2.0)
    ary.append(gsl_vector_get(eval,1) - 1.0)
    ary.append(gsl_vector_get(eval,2) - 3.0)
    gsl_matrix_free(a_diag)
    gsl_matrix_free(a)
    gsl_matrix_free(h)
    gsl_matrix_free(c)
    gsl_vector_free(eval)
    gsl_eigen_symm_free(w)
    return ary

def t_gsl_eigen_symmv():
    cdef gsl_matrix *a_diag, *a, *h, *c, *evec, *evec1
    cdef gsl_vector *eval, *v, *v1, *eval1
    cdef gsl_vector_view vw
    cdef gsl_eigen_symmv_workspace *w
    a_diag = gsl_matrix_calloc(3,3)
    a = gsl_matrix_alloc(3,3)
    evec = gsl_matrix_alloc(3,3)
    evec1 = gsl_matrix_alloc(3,3)
    c = gsl_matrix_alloc(3,3)
    h = gsl_matrix_alloc(3,3)
    eval = gsl_vector_alloc(3)
    eval1 = gsl_vector_alloc(3)
    v1 = gsl_vector_alloc(3)
    w = gsl_eigen_symmv_alloc(3)
    for i in range(3):
        gsl_matrix_set(a_diag, i, i, i+1)
    Nh = array([[0.6, -0.8,0],[-0.8, -0.6,0],[0,0,1]])
    gsl_matrix_set_from_Num(h,Nh)
    # a_diag*h
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans,1.0, a_diag, h, 0.0, c)
    # h^T*a_diag*h
    gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, h, c, 0.0, a)
    gsl_eigen_symmv(a, eval, evec, w)
    ary = []
    for i in range(3):
        vw = gsl_matrix_column(evec,i)
        v = &vw.vector
        gsl_vector_memcpy(v1,v)
        gsl_blas_dgemv(CblasNoTrans, 1.0, a, v, -gsl_vector_get(eval,i), v1)
        for j in range(3):
            ary.append(gsl_vector_get(v1,j))

    # gsl_eigen_symmv_sort
    Neval = array([2,1,3])
    for i in range(3):
        ary.append(gsl_vector_get(eval,i) - Neval[i])

    gsl_matrix_memcpy(evec1, evec)
    gsl_vector_memcpy(eval1, eval)
    gsl_eigen_symmv_sort(eval1, evec1, GSL_EIGEN_SORT_VAL_ASC)
    Neval = array([1,2,3])
    for i in range(3):
        ary.append(gsl_vector_get(eval1,i) - Neval[i])

    # evec[0] == evec1[1]
    vw = gsl_matrix_column(evec,0)
    v = &vw.vector
    gsl_vector_memcpy(v1,v)
    vw = gsl_matrix_column(evec1,1)
    v = &vw.vector
    gsl_vector_sub(v1,v)
    for i in range(3):
        ary.append(gsl_vector_get(v1,i))

    # evec[1] == evec1[0]
    vw = gsl_matrix_column(evec,1)
    v = &vw.vector
    gsl_vector_memcpy(v1,v)
    vw = gsl_matrix_column(evec1,0)
    v = &vw.vector
    gsl_vector_sub(v1,v)
    for i in range(3):
        ary.append(gsl_vector_get(v1,i))

    gsl_matrix_free(a_diag)
    gsl_matrix_free(a)
    gsl_matrix_free(h)
    gsl_matrix_free(c)
    gsl_matrix_free(evec)
    gsl_matrix_free(evec1)
    gsl_vector_free(eval)
    gsl_vector_free(eval1)
    gsl_vector_free(v1)
    gsl_eigen_symmv_free(w)
    return ary

def t_gsl_eigen_herm():
    cdef gsl_matrix_complex *a
    cdef gsl_vector *eval
    cdef gsl_eigen_herm_workspace *w
    eval = gsl_vector_alloc(2)
    a = gsl_matrix_complex_alloc(2,2)
    Na = array([[0, 1j],[-1j,0]])
    gsl_matrix_complex_set_from_Num(a,Na)
    w = gsl_eigen_herm_alloc(2)
    gsl_eigen_herm(a, eval, w)
    ary = []
    Neval = array([1,-1])
    for i in range(2):
        ary.append(gsl_vector_get(eval,i) - Neval[i])
    gsl_matrix_complex_free(a)
    gsl_vector_free(eval)
    gsl_eigen_herm_free(w)
    return ary

def t_gsl_eigen_hermv():
    cdef gsl_matrix_complex *a, *evec, *ac
    cdef gsl_vector *eval
    cdef gsl_vector_complex *v, *v1
    cdef gsl_eigen_hermv_workspace *w
    cdef gsl_vector_complex_view vw
    cdef gsl_complex z
    cdef double d1
    eval = gsl_vector_alloc(2)
    a = gsl_matrix_complex_alloc(2,2)
    ac = gsl_matrix_complex_alloc(2,2)
    evec = gsl_matrix_complex_alloc(2,2)
    v1 = gsl_vector_complex_alloc(2)
    Na = array([[0, 1j],[-1j,0]])
    gsl_matrix_complex_set_from_Num(a,Na)
    w = gsl_eigen_hermv_alloc(2)
    gsl_matrix_complex_memcpy(ac,a)
    gsl_eigen_hermv(ac, eval, evec, w)
    ary = []
    Neval = array([1,-1])
    for i in range(2):
        ary.append(gsl_vector_get(eval,i) - Neval[i])
    for i in range(2):
        vw = gsl_matrix_complex_column(evec,i)
        v = &vw.vector
        gsl_vector_complex_memcpy(v1,v)
        GSL_SET_COMPLEX(&z,-gsl_vector_get(eval,i),0)
        gsl_blas_zgemv(CblasNoTrans, GSL_COMPLEX_ONE, a, v, z, v1)
        for j in range(2):
            z = gsl_vector_complex_get(v1,j)
            ary.extend([GSL_REAL(z), GSL_IMAG(z)])

    gsl_matrix_complex_free(a)
    gsl_matrix_complex_free(ac)
    gsl_matrix_complex_free(evec)
    gsl_vector_free(eval)
    gsl_eigen_hermv_free(w)
    return ary
