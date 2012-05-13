from cython_gsl cimport *
from numpy import array, dot, transpose, outer, conjugate
from numpy.linalg import inv

def t_gsl_blas_ddot():
    cdef gsl_vector *v1, *v2
    v1 = gsl_vector_alloc(2)
    v2 = gsl_vector_alloc(2)
    gsl_vector_set_basis(v1, 0)
    gsl_vector_set_basis(v2, 1)
    cdef double res
    gsl_blas_ddot(v1, v2, &res)
    ary = []
    ary.append(res)
    gsl_blas_ddot(v1, v1, &res)
    ary.append(res - 1)
    gsl_vector_free(v1)
    gsl_vector_free(v2)
    return ary

def t_gsl_blas_zdotu():
    cdef gsl_complex z
    cdef gsl_vector_complex *v1, *v2
    v1 = gsl_vector_complex_alloc(2)
    v2 = gsl_vector_complex_alloc(2)
    gsl_vector_complex_set(v1, 0, gsl_complex_rect(1,0))
    gsl_vector_complex_set(v1, 1, gsl_complex_rect(0,1))
    gsl_vector_complex_set(v2, 0, gsl_complex_rect(3,0))
    gsl_vector_complex_set(v2, 1, gsl_complex_rect(0,2))
    # v1^T v2
    gsl_blas_zdotu(v1, v2, &z)
    ary = []
    ary.extend([GSL_REAL(z) - 1, GSL_IMAG(z)])
    # v1^H v2
    gsl_blas_zdotc(v1, v2, &z)
    ary.extend([GSL_REAL(z) - 5, GSL_IMAG(z)])
    gsl_vector_complex_free(v1)
    gsl_vector_complex_free(v2)
    return ary

def t_gsl_blas_dnrm2():
    cdef gsl_vector *v1, *v2
    v1 = gsl_vector_alloc(2)
    gsl_vector_set(v1,0, 1.2)
    gsl_vector_set(v1,1, 1.0)
    ary = [gsl_blas_dnrm2(v1) - sqrt(2.44)]
    gsl_vector_free(v1)
    return ary

def t_gsl_blas_dasum():
    cdef gsl_vector *v1, *v2
    v1 = gsl_vector_alloc(2)
    gsl_vector_set(v1,0, 1.2)
    gsl_vector_set(v1,1, -1.0)
    ary = [gsl_blas_dasum(v1) - 2.2]
    ary.append(gsl_blas_idamax(v1) - 0)
    gsl_vector_free(v1)
    return ary

def t_gsl_blas_dznrm2():
    cdef gsl_vector_complex *v1
    v1 = gsl_vector_complex_alloc(2)
    gsl_vector_complex_set(v1, 0, gsl_complex_rect(0.3,0))
    gsl_vector_complex_set(v1, 1, gsl_complex_rect(0,0.4))
    ary = [gsl_blas_dznrm2(v1) - 0.5]
    gsl_vector_complex_free(v1)
    return ary

def t_gsl_blas_dzasum():
    cdef gsl_vector_complex *v1
    v1 = gsl_vector_complex_alloc(2)
    gsl_vector_complex_set(v1, 0, gsl_complex_rect(0.3,1))
    gsl_vector_complex_set(v1, 1, gsl_complex_rect(1,-0.4))
    cdef double d1
    d1 = 0
    for i from 0 <= i < v1.size:
        d1 = d1 + fabs(GSL_REAL(gsl_vector_complex_get(v1,i)))
        d1 = d1 + fabs(GSL_IMAG(gsl_vector_complex_get(v1,i)))
    ary = [gsl_blas_dzasum(v1) - d1]
    ary.append(gsl_blas_izamax(v1) - 1)
    gsl_vector_complex_free(v1)
    return ary

def t_gsl_blas_dswap():
    cdef gsl_vector *v1, *v2
    v1 = gsl_vector_alloc(2)
    v2 = gsl_vector_alloc(2)
    gsl_vector_set_basis(v1, 0)
    gsl_vector_set_basis(v2, 1)
    gsl_blas_dswap(v2,v1)
    ary = []
    ary.append(gsl_vector_get(v1,0))
    ary.append(gsl_vector_get(v1,1) - 1)
    ary.append(gsl_vector_get(v2,0) - 1)
    ary.append(gsl_vector_get(v2,1))
    gsl_vector_free(v1)
    gsl_vector_free(v2)
    return ary

def t_gsl_blas_dcopy():
    cdef gsl_vector *v1, *v2
    v1 = gsl_vector_alloc(2)
    v2 = gsl_vector_alloc(2)
    gsl_vector_set_basis(v1, 0)
    gsl_vector_set_basis(v2, 1)
    gsl_blas_dcopy(v1,v2)
    ary = []
    ary.append(gsl_vector_get(v1,0) - 1)
    ary.append(gsl_vector_get(v1,1))
    ary.append(gsl_vector_get(v2,0) - 1)
    ary.append(gsl_vector_get(v2,1))
    gsl_vector_free(v1)
    gsl_vector_free(v2)
    return ary

def t_gsl_blas_daxpy():
    cdef gsl_vector *v1, *v2
    v1 = gsl_vector_alloc(2)
    v2 = gsl_vector_alloc(2)
    gsl_vector_set_basis(v1, 0)
    gsl_vector_set_basis(v2, 1)
    # v2 = 0.5*v1 + v2
    gsl_blas_daxpy(0.5, v1,v2)
    ary = []
    ary.append(gsl_vector_get(v2,0) - 0.5)
    ary.append(gsl_vector_get(v2,1) - 1)
    gsl_vector_free(v1)
    gsl_vector_free(v2)
    return ary

def t_gsl_blas_zswap():
    cdef gsl_vector_complex *v1, *v2
    cdef gsl_complex z
    v1 = gsl_vector_complex_alloc(2)
    v2 = gsl_vector_complex_alloc(2)
    gsl_vector_complex_set(v1, 0, gsl_complex_rect(1,1))
    gsl_vector_complex_set(v1, 1, gsl_complex_rect(2,2))
    gsl_vector_complex_set(v2, 0, gsl_complex_rect(3,4))
    gsl_vector_complex_set(v2, 1, gsl_complex_rect(5,6))
    gsl_blas_zswap(v1,v2)
    ary = []
    z = gsl_vector_complex_get(v1,0)
    ary.extend([GSL_REAL(z) - 3, GSL_IMAG(z) - 4])
    z = gsl_vector_complex_get(v1,1)
    ary.extend([GSL_REAL(z) - 5, GSL_IMAG(z) - 6])
    z = gsl_vector_complex_get(v2,0)
    ary.extend([GSL_REAL(z) - 1, GSL_IMAG(z) - 1])
    z = gsl_vector_complex_get(v2,1)
    ary.extend([GSL_REAL(z) - 2, GSL_IMAG(z) - 2])
    gsl_vector_complex_free(v1)
    gsl_vector_complex_free(v2)
    return ary

def t_gsl_blas_zcopy():
    cdef gsl_vector_complex *v1, *v2
    cdef gsl_complex z
    v1 = gsl_vector_complex_alloc(2)
    v2 = gsl_vector_complex_alloc(2)
    gsl_vector_complex_set(v1, 0, gsl_complex_rect(1,1))
    gsl_vector_complex_set(v1, 1, gsl_complex_rect(2,2))
    gsl_vector_complex_set(v2, 0, gsl_complex_rect(3,4))
    gsl_vector_complex_set(v2, 1, gsl_complex_rect(5,6))
    gsl_blas_zcopy(v1,v2)
    ary = []
    z = gsl_vector_complex_get(v1,0)
    ary.extend([GSL_REAL(z) - 1, GSL_IMAG(z) - 1])
    z = gsl_vector_complex_get(v1,1)
    ary.extend([GSL_REAL(z) - 2, GSL_IMAG(z) - 2])
    z = gsl_vector_complex_get(v2,0)
    ary.extend([GSL_REAL(z) - 1, GSL_IMAG(z) - 1])
    z = gsl_vector_complex_get(v2,1)
    ary.extend([GSL_REAL(z) - 2, GSL_IMAG(z) - 2])
    gsl_vector_complex_free(v1)
    gsl_vector_complex_free(v2)
    return ary

def t_gsl_blas_zaxpy():
    cdef gsl_vector_complex *v1, *v2
    cdef gsl_complex z
    v1 = gsl_vector_complex_alloc(2)
    v2 = gsl_vector_complex_alloc(2)
    gsl_vector_complex_set(v1, 0, gsl_complex_rect(1,1))
    gsl_vector_complex_set(v1, 1, gsl_complex_rect(2,2))
    gsl_vector_complex_set(v2, 0, gsl_complex_rect(3,4))
    gsl_vector_complex_set(v2, 1, gsl_complex_rect(5,6))
    gsl_blas_zaxpy(gsl_complex_rect(0.5, 0.6), v1, v2)
    # v2 = alpha*v1 + v2
    ary = []
    z = gsl_vector_complex_get(v1,0)
    ary.extend([GSL_REAL(z) - 1, GSL_IMAG(z) - 1])
    z = gsl_vector_complex_get(v2,0)
    w = (1 +1j)*(0.5 + 0.6j) + (3 + 4j)
    ary.extend([GSL_REAL(z) - w.real, GSL_IMAG(z) - w.imag])
    w = (2 + 2j)*(0.5 + 0.6j) + (5 + 6j)
    z = gsl_vector_complex_get(v2,1)
    ary.extend([GSL_REAL(z) - w.real, GSL_IMAG(z) - w.imag])
    gsl_vector_complex_free(v1)
    gsl_vector_complex_free(v2)
    return ary

def t_gsl_blas_drotg():
    cdef double a, b, c, s, r
    a = 1.0; b = 2.0
    r = sqrt(a*a + b*b)
    gsl_blas_drotg(&a, &b, &c, &s)
    #c = a/r; s = b/r; r = sqrt(a^2 + b^2)
    # a, b have been rewritten, so don't use them:
    ary = [c - 1.0/r, s - 2.0/r]
    return ary


def t_gsl_blas_dscal():
    cdef gsl_vector * v1
    v1 = gsl_vector_alloc(2)
    gsl_vector_set(v1,0, 1.1)
    gsl_vector_set(v1,1, 2.1)
    gsl_blas_dscal(0.5, v1)
    ary = []
    ary.append(gsl_vector_get(v1,0) - 1.1*0.5)
    ary.append(gsl_vector_get(v1,1) - 2.1*0.5)
    gsl_vector_free(v1)
    return ary

def t_gsl_blas_zscal():
    cdef gsl_vector_complex * v1
    v1 = gsl_vector_complex_alloc(2)
    gsl_vector_complex_set(v1,0, gsl_complex_rect(1.1, 2.1))
    gsl_vector_complex_set(v1,1, gsl_complex_rect(3.1, 4.1))
    gsl_blas_zscal(gsl_complex_rect(5.1, 6.1), v1)
    cdef gsl_complex z
    ary = []
    z = gsl_vector_complex_get(v1, 0)
    w = (1.1 + 2.1j)*(5.1 + 6.1j)
    ary.append(GSL_REAL(z) - w.real)
    ary.append(GSL_IMAG(z) - w.imag)
    z = gsl_vector_complex_get(v1, 1)
    w = (3.1 + 4.1j)*(5.1 + 6.1j)
    ary.append(GSL_REAL(z) - w.real)
    ary.append(GSL_IMAG(z) - w.imag)
    gsl_vector_complex_free(v1)
    return ary


def t_gsl_blas_zdscal():
    cdef gsl_vector_complex * v1
    v1 = gsl_vector_complex_alloc(2)
    gsl_vector_complex_set(v1,0, gsl_complex_rect(1.1, 2.1))
    gsl_vector_complex_set(v1,1, gsl_complex_rect(3.1, 4.1))
    gsl_blas_zdscal(0.5, v1)
    cdef gsl_complex z
    ary = []
    z = gsl_vector_complex_get(v1, 0)
    w = (1.1 + 2.1j)*0.5
    ary.append(GSL_REAL(z) - w.real)
    ary.append(GSL_IMAG(z) - w.imag)
    z = gsl_vector_complex_get(v1, 1)
    w = (3.1 + 4.1j)*0.5
    ary.append(GSL_REAL(z) - w.real)
    ary.append(GSL_IMAG(z) - w.imag)
    gsl_vector_complex_free(v1)
    return ary

def t_gsl_blas_dgemv():
    cdef gsl_matrix * m1
    cdef gsl_vector * v1, * v2
    m1 = gsl_matrix_alloc(2,2)
    v1 = gsl_vector_alloc(2)
    v2 = gsl_vector_alloc(2)
    gsl_vector_set_basis(v1,0)
    gsl_vector_set_basis(v2,1)
    Nm1 = array([[2,4], [1,3]])
    Nv1 = array([1,0])
    Nv2 = array([0,1])
    alpha = 2
    beta = 3
    ary = []
    for i in range(2):
        for j in range(2):
            gsl_matrix_set(m1, i, j, Nm1[i,j])

    # v2 = m1 * v1 * alpha + v2 * beta
    Nv3 = dot(Nm1,Nv1) * alpha + Nv2 * beta
    gsl_blas_dgemv(CblasNoTrans, alpha, m1, v1, beta, v2)
    for i in range(2):
        ary.append(gsl_vector_get(v2,i) - Nv3[i])

    # v2 = m1^T * v1 * alpha + v2 * beta
    gsl_vector_set_basis(v2,1)
    Nv3 = dot(transpose(Nm1),Nv1) * alpha + Nv2 * beta
    gsl_blas_dgemv(CblasTrans, alpha, m1, v1, beta, v2)
    for i in range(2):
        ary.append(gsl_vector_get(v2,i) - Nv3[i])
    gsl_vector_free(v1)
    gsl_vector_free(v2)
    gsl_matrix_free(m1)
    return ary

def t_gsl_blas_dtrmv():
    cdef gsl_matrix * m1
    cdef gsl_vector * v1
    m1 = gsl_matrix_alloc(3,3)
    v1 = gsl_vector_alloc(3)
    ary = []
    '''
    CBLAS_UPLO_t = CblasUpper, CblasLower
    CBLAS_TRANSPOSE_t = CblasNoTrans, CblasTrans
    CBLAS_DIAG_t = CblasNonUnit, CblasUpper
    There are 2**3 = 8 cases
    '''

    # upper triangular matrix, with non-unit diagonal
    Nm1 = array([[1,2,3], [0,1,1], [0,0,2]])
    Nv1 = array([3,2,4])
    for i in range(3):
        for j in range(3):
            gsl_matrix_set(m1,i,j, Nm1[i,j])
    for i in range(3):
        gsl_vector_set(v1, i, Nv1[i])

    # 1) v1 = m1 * v1
    Nv2 = dot(Nm1, Nv1)
    gsl_blas_dtrmv(CblasUpper, CblasNoTrans, CblasNonUnit, m1, v1)
    for i in range(3):
        ary.append(gsl_vector_get(v1, i) - Nv2[i])

    # 2) v1 = m1^T * v1
    for i in range(3):
        gsl_vector_set(v1, i, Nv1[i])
    Nv2 = dot(transpose(Nm1), Nv1)
    gsl_blas_dtrmv(CblasUpper, CblasTrans, CblasNonUnit, m1, v1)
    for i in range(3):
        ary.append(gsl_vector_get(v1, i) - Nv2[i])

    # upper triangular matrix, with unit diagonal
    Nm1 = array([[1,2,3], [0,1,1], [0,0,1]])
    Nv1 = array([3,2,4])
    for i in range(3):
        for j in range(3):
            gsl_matrix_set(m1,i,j, Nm1[i,j])
    for i in range(3):
        gsl_vector_set(v1, i, Nv1[i])

    # 3) v1 = m1 * v1
    Nv2 = dot(Nm1, Nv1)
    gsl_blas_dtrmv(CblasUpper, CblasNoTrans, CblasUnit, m1, v1)
    for i in range(3):
        ary.append(gsl_vector_get(v1, i) - Nv2[i])

    # 4) v1 = m1^T * v1
    for i in range(3):
        gsl_vector_set(v1, i, Nv1[i])
    Nv2 = dot(transpose(Nm1), Nv1)
    gsl_blas_dtrmv(CblasUpper, CblasTrans, CblasUnit, m1, v1)
    for i in range(3):
        ary.append(gsl_vector_get(v1, i) - Nv2[i])

    ############################################
    # lower triangular matrix, with non-unit diagonal
    Nm1 = array([[1,0,0], [2,1,0], [1,2,2]])
    Nv1 = array([3,2,4])
    for i in range(3):
        for j in range(3):
            gsl_matrix_set(m1,i,j, Nm1[i,j])
    for i in range(3):
        gsl_vector_set(v1, i, Nv1[i])

    # 5) v1 = m1 * v1
    Nv2 = dot(Nm1, Nv1)
    gsl_blas_dtrmv(CblasLower, CblasNoTrans, CblasNonUnit, m1, v1)
    for i in range(3):
        ary.append(gsl_vector_get(v1, i) - Nv2[i])

    # 6) v1 = m1^T * v1
    for i in range(3):
        gsl_vector_set(v1, i, Nv1[i])
    Nv2 = dot(transpose(Nm1), Nv1)
    gsl_blas_dtrmv(CblasLower, CblasTrans, CblasNonUnit, m1, v1)
    for i in range(3):
        ary.append(gsl_vector_get(v1, i) - Nv2[i])

    # lower triangular matrix, with unit diagonal
    Nm1 = array([[1,0,0], [2,1,0], [1,2,1]])
    Nv1 = array([3,2,4])
    for i in range(3):
        for j in range(3):
            gsl_matrix_set(m1,i,j, Nm1[i,j])
    for i in range(3):
        gsl_vector_set(v1, i, Nv1[i])

    # 7) v1 = m1 * v1
    Nv2 = dot(Nm1, Nv1)
    gsl_blas_dtrmv(CblasLower, CblasNoTrans, CblasUnit, m1, v1)
    for i in range(3):
        ary.append(gsl_vector_get(v1, i) - Nv2[i])

    # 8) v1 = m1^T * v1
    for i in range(3):
        gsl_vector_set(v1, i, Nv1[i])
    Nv2 = dot(transpose(Nm1), Nv1)
    gsl_blas_dtrmv(CblasLower, CblasTrans, CblasUnit, m1, v1)
    for i in range(3):
        ary.append(gsl_vector_get(v1, i) - Nv2[i])

    gsl_vector_free(v1)
    gsl_matrix_free(m1)
    return ary

def t_gsl_blas_dtrsv():
    cdef gsl_matrix * m1
    cdef gsl_vector * v1
    m1 = gsl_matrix_alloc(3,3)
    v1 = gsl_vector_alloc(3)
    ary = []
    '''
    CBLAS_UPLO_t = CblasUpper, CblasLower
    CBLAS_TRANSPOSE_t = CblasNoTrans, CblasTrans
    CBLAS_DIAG_t = CblasNonUnit, CblasUpper
    There are 2**3 = 8 cases
    '''

    # upper triangular matrix, with non-unit diagonal
    Nm1 = array([[1,2,3], [0,1,1], [0,0,2]])
    Nv1 = array([3,2,4])
    for i in range(3):
        for j in range(3):
            gsl_matrix_set(m1,i,j, Nm1[i,j])
    for i in range(3):
        gsl_vector_set(v1, i, Nv1[i])

    # 1) v1 = m1 * v1
    Nv2 = dot(inv(Nm1), Nv1)
    gsl_blas_dtrsv(CblasUpper, CblasNoTrans, CblasNonUnit, m1, v1)
    for i in range(3):
        ary.append(gsl_vector_get(v1, i) - Nv2[i])

    # 2) v1 = m1^T * v1
    for i in range(3):
        gsl_vector_set(v1, i, Nv1[i])
    Nv2 = dot(inv(transpose(Nm1)), Nv1)
    gsl_blas_dtrsv(CblasUpper, CblasTrans, CblasNonUnit, m1, v1)
    for i in range(3):
        ary.append(gsl_vector_get(v1, i) - Nv2[i])

    # upper triangular matrix, with unit diagonal
    Nm1 = array([[1,2,3], [0,1,1], [0,0,1]])
    Nv1 = array([3,2,4])
    for i in range(3):
        for j in range(3):
            gsl_matrix_set(m1,i,j, Nm1[i,j])
    for i in range(3):
        gsl_vector_set(v1, i, Nv1[i])

    # 3) v1 = m1 * v1
    Nv2 = dot(inv(Nm1), Nv1)
    gsl_blas_dtrsv(CblasUpper, CblasNoTrans, CblasUnit, m1, v1)
    for i in range(3):
        ary.append(gsl_vector_get(v1, i) - Nv2[i])

    # 4) v1 = m1^T * v1
    for i in range(3):
        gsl_vector_set(v1, i, Nv1[i])
    Nv2 = dot(inv(transpose(Nm1)), Nv1)
    gsl_blas_dtrsv(CblasUpper, CblasTrans, CblasUnit, m1, v1)
    for i in range(3):
        ary.append(gsl_vector_get(v1, i) - Nv2[i])

    ############################################
    # lower triangular matrix, with non-unit diagonal
    Nm1 = array([[1,0,0], [2,1,0], [1,2,2]])
    Nv1 = array([3,2,4])
    for i in range(3):
        for j in range(3):
            gsl_matrix_set(m1,i,j, Nm1[i,j])
    for i in range(3):
        gsl_vector_set(v1, i, Nv1[i])

    # 5) v1 = m1 * v1
    Nv2 = dot(inv(Nm1), Nv1)
    gsl_blas_dtrsv(CblasLower, CblasNoTrans, CblasNonUnit, m1, v1)
    for i in range(3):
        ary.append(gsl_vector_get(v1, i) - Nv2[i])

    # 6) v1 = m1^T * v1
    for i in range(3):
        gsl_vector_set(v1, i, Nv1[i])
    Nv2 = dot(inv(transpose(Nm1)), Nv1)
    gsl_blas_dtrsv(CblasLower, CblasTrans, CblasNonUnit, m1, v1)
    for i in range(3):
        ary.append(gsl_vector_get(v1, i) - Nv2[i])

    # lower triangular matrix, with unit diagonal
    Nm1 = array([[1,0,0], [2,1,0], [1,2,1]])
    Nv1 = array([3,2,4])
    for i in range(3):
        for j in range(3):
            gsl_matrix_set(m1,i,j, Nm1[i,j])
    for i in range(3):
        gsl_vector_set(v1, i, Nv1[i])

    # 7) v1 = m1 * v1
    Nv2 = dot(inv(Nm1), Nv1)
    gsl_blas_dtrsv(CblasLower, CblasNoTrans, CblasUnit, m1, v1)
    for i in range(3):
        ary.append(gsl_vector_get(v1, i) - Nv2[i])

    # 8) v1 = m1^T * v1
    for i in range(3):
        gsl_vector_set(v1, i, Nv1[i])
    Nv2 = dot(inv(transpose(Nm1)), Nv1)
    gsl_blas_dtrsv(CblasLower, CblasTrans, CblasUnit, m1, v1)
    for i in range(3):
        ary.append(gsl_vector_get(v1, i) - Nv2[i])

    gsl_vector_free(v1)
    gsl_matrix_free(m1)
    return ary

def t_gsl_blas_dsymv():
    cdef gsl_matrix * m1
    cdef gsl_vector * v1, * v2
    m1 = gsl_matrix_calloc(3,3)
    v1 = gsl_vector_alloc(3)
    v2 = gsl_vector_alloc(3)
    ary = []
    Nm1 = array([[1,2,3], [2,1,1], [3,1,2]])
    Nv1 = array([3,2,4])
    Nv2 = array([5,7,6])
    alpha = 0.5
    beta = 0.6
    Nv3 = alpha * dot(Nm1, Nv1) + beta * Nv2
    for i in range(0,3):
        gsl_vector_set(v1,i, Nv1[i])
        gsl_vector_set(v2,i, Nv2[i])

    # CBLAS_UPLO_t = CblasUpper, CblasLower

    # upper matrix
    for i in range(0,3):
        for j in range(i, 3):
            gsl_matrix_set(m1, i,j,Nm1[i,j])

    gsl_blas_dsymv(CblasUpper, alpha, m1, v1, beta, v2)
    for i in range(0,3):
        ary.append(gsl_vector_get(v2,i) - Nv3[i])

    # lower matrix
    for i in range(0,3):
        gsl_vector_set(v2,i, Nv2[i])
    gsl_matrix_set_zero(m1)
    for i in range(0,3):
        for j in range(0, i+1):
            gsl_matrix_set(m1, i,j,Nm1[i,j])

    gsl_blas_dsymv(CblasLower, alpha, m1, v1, beta, v2)
    for i in range(0,3):
        ary.append(gsl_vector_get(v2,i) - Nv3[i])
    gsl_vector_free(v1)
    gsl_vector_free(v2)
    gsl_matrix_free(m1)
    return ary

def t_gsl_blas_dger():
    cdef gsl_matrix * m1
    cdef gsl_vector * v1, * v2
    m1 = gsl_matrix_calloc(3,3)
    v1 = gsl_vector_alloc(3)
    v2 = gsl_vector_alloc(3)
    ary = []
    Nm1 = array([[1,2,3], [2,1,1], [3,1,2]])
    Nv1 = array([3,2,4])
    Nv2 = array([5,7,6])
    alpha = 0.5
    Nm2 = alpha * outer(Nv1, Nv2) + Nm1
    for i in range(0,3):
        gsl_vector_set(v1,i, Nv1[i])
        gsl_vector_set(v2,i, Nv2[i])
    for i in range(0,3):
        for j in range(0, 3):
            gsl_matrix_set(m1, i,j,Nm1[i,j])
    gsl_blas_dger(alpha, v1, v2, m1)
    for i in range(0,3):
        for j in range(0, 3):
            ary.append(gsl_matrix_get(m1, i,j) - Nm2[i,j])
    gsl_vector_free(v1)
    gsl_vector_free(v2)
    gsl_matrix_free(m1)
    return ary


def t_gsl_blas_dsyr():
    cdef gsl_matrix * m1
    cdef gsl_vector * v1
    m1 = gsl_matrix_calloc(3,3)
    v1 = gsl_vector_alloc(3)
    ary = []
    Nm1 = array([[1,2,3], [2,1,1], [3,1,2]])
    Nv1 = array([3,2,4])
    alpha = 0.5
    Nm2 = alpha * outer(Nv1, Nv1) + Nm1
    for i in range(0,3):
        gsl_vector_set(v1,i, Nv1[i])

    # upper matrix
    for i in range(0,3):
        for j in range(i, 3):
            gsl_matrix_set(m1, i,j,Nm1[i,j])
    gsl_blas_dsyr(CblasUpper, alpha, v1, m1)
    for i in range(0,3):
        for j in range(i, 3):
            ary.append(gsl_matrix_get(m1, i,j) - Nm2[i,j])
    for i in range(0,3):
        for j in range(0, i):
            ary.append(gsl_matrix_get(m1, i,j))

    # lower matrix
    gsl_matrix_set_zero(m1)
    for i in range(0,3):
        for j in range(0, i + 1):
            gsl_matrix_set(m1, i,j,Nm1[i,j])
    gsl_blas_dsyr(CblasLower, alpha, v1, m1)
    for i in range(0,3):
        for j in range(0, i + 1):
            ary.append(gsl_matrix_get(m1, i,j) - Nm2[i,j])
    for i in range(0,3):
        for j in range(i+1, 3):
            ary.append(gsl_matrix_get(m1, i,j))
    gsl_vector_free(v1)
    gsl_matrix_free(m1)
    return ary


def t_gsl_blas_dsyr2():
    cdef gsl_matrix * m1
    cdef gsl_vector * v1, * v2
    m1 = gsl_matrix_calloc(3,3)
    v1 = gsl_vector_alloc(3)
    v2 = gsl_vector_alloc(3)
    ary = []
    Nm1 = array([[1,2,3], [2,1,1], [3,1,2]])
    Nv1 = array([3,2,4])
    Nv2 = array([5,7,6])
    alpha = 0.5
    Nm2 = alpha * outer(Nv1, Nv2) + alpha * outer(Nv2, Nv1) + Nm1
    for i in range(0,3):
        gsl_vector_set(v1,i, Nv1[i])
        gsl_vector_set(v2,i, Nv2[i])

    # upper matrix
    for i in range(0,3):
        for j in range(i, 3):
            gsl_matrix_set(m1, i,j,Nm1[i,j])
    gsl_blas_dsyr2(CblasUpper, alpha, v1, v2, m1)
    for i in range(0,3):
        for j in range(i, 3):
            ary.append(gsl_matrix_get(m1, i,j) - Nm2[i,j])
    for i in range(0,3):
        for j in range(0, i):
            ary.append(gsl_matrix_get(m1, i,j))

    # lower matrix
    gsl_matrix_set_zero(m1)
    for i in range(0,3):
        for j in range(0, i + 1):
            gsl_matrix_set(m1, i,j,Nm1[i,j])
    gsl_blas_dsyr2(CblasLower, alpha, v1, v2, m1)
    for i in range(0,3):
        for j in range(0, i + 1):
            ary.append(gsl_matrix_get(m1, i,j) - Nm2[i,j])
    for i in range(0,3):
        for j in range(i+1, 3):
            ary.append(gsl_matrix_get(m1, i,j))

    gsl_vector_free(v1)
    gsl_matrix_free(m1)
    return ary

def t_gsl_blas_zgemv():
    cdef gsl_complex z, alpha, beta
    cdef gsl_matrix_complex * m1
    cdef gsl_vector_complex * v1, * v2
    m1 = gsl_matrix_complex_alloc(2,2)
    v1 =  gsl_vector_complex_alloc(2)
    v2 =  gsl_vector_complex_alloc(2)
    Nm1 = array([[1+2j, 2+3j],[5+6j, 6+7j]])
    Nv1 = array([7+1j, 3+2j])
    Nv2 = array([1+2j, 3+1j])
    Nalpha = 0.1 + 0.2j
    Nbeta = 0.3 + 0.4j
    alpha = gsl_complex_rect(Nalpha.real, Nalpha.imag)
    beta = gsl_complex_rect(Nbeta.real, Nbeta.imag)

    ary = []
    for i in range(2):
        for j in range(2):
            z = gsl_complex_rect(Nm1[i,j].real, Nm1[i,j].imag)
            gsl_matrix_complex_set(m1, i, j, z)
    for i in range(2):
        z = gsl_complex_rect(Nv1[i].real, Nv1[i].imag)
        gsl_vector_complex_set(v1, i, z)
        z = gsl_complex_rect(Nv2[i].real, Nv2[i].imag)
        gsl_vector_complex_set(v2, i, z)

    # CBLAS_TRANSPOSE_t = CblasNoTrans, CblasTrans, CblasConjTrans
    # 3 cases
    # 1) alpha * m1 * v1 + beta * v2
    Nv3 = Nalpha * dot(Nm1,Nv1) + Nbeta * Nv2
    gsl_blas_zgemv(CblasNoTrans, alpha, m1, v1, beta, v2)
    for i in range(2):
        z = gsl_vector_complex_get(v2,i)
        ary.append(GSL_REAL(z) - Nv3[i].real)
        ary.append(GSL_IMAG(z) - Nv3[i].imag)

    # 2) alpha * m1^T * v1 + beta * v2
    Nv3 = Nalpha * dot(transpose(Nm1),Nv1) + Nbeta * Nv2
    for i in range(2):
        z = gsl_complex_rect(Nv2[i].real, Nv2[i].imag)
        gsl_vector_complex_set(v2, i, z)
    gsl_blas_zgemv(CblasTrans, alpha, m1, v1, beta, v2)
    for i in range(2):
        z = gsl_vector_complex_get(v2,i)
        ary.append(GSL_REAL(z) - Nv3[i].real)
        ary.append(GSL_IMAG(z) - Nv3[i].imag)

    # 3) alpha * m1^H * v1 + beta * v2
    Nv3 = Nalpha * dot(conjugate(transpose(Nm1)),Nv1) + Nbeta * Nv2
    for i in range(2):
        z = gsl_complex_rect(Nv2[i].real, Nv2[i].imag)
        gsl_vector_complex_set(v2, i, z)
    gsl_blas_zgemv(CblasConjTrans, alpha, m1, v1, beta, v2)
    for i in range(2):
        z = gsl_vector_complex_get(v2,i)
        ary.append(GSL_REAL(z) - Nv3[i].real)
        ary.append(GSL_IMAG(z) - Nv3[i].imag)

    gsl_matrix_complex_free(m1)
    gsl_vector_complex_free(v1)
    gsl_vector_complex_free(v2)
    return ary

def t_gsl_blas_ztrmv():
    cdef gsl_complex z
    cdef gsl_matrix_complex * m1
    cdef gsl_vector_complex * v1
    m1 = gsl_matrix_complex_alloc(2,2)
    v1 =  gsl_vector_complex_alloc(2)
    ary = []
    '''
    CBLAS_UPLO_t = CblasUpper, CblasLower
    CBLAS_TRANSPOSE_t = CblasNoTrans, CblasTrans, CblasConjTrans
    CBLAS_DIAG_t = CblasNonUnit, CblasUpper
    There are 2*3*2 = 12 cases
    Here I check only the 6 cases with upper triangular matrix
    '''

    # upper triangular matrix, with non-unit diagonal
    Nm1 = array([[1+2j, 2+3j],[0, 6+7j]])
    Nv1 = array([7+1j, 3+2j])
    for i in range(2):
        for j in range(2):
            z = gsl_complex_rect(Nm1[i,j].real, Nm1[i,j].imag)
            gsl_matrix_complex_set(m1, i, j, z)
    for i in range(2):
        z = gsl_complex_rect(Nv1[i].real, Nv1[i].imag)
        gsl_vector_complex_set(v1, i, z)

    # 1) v1 = m1 * v1
    Nv2 = dot(Nm1, Nv1)
    gsl_blas_ztrmv(CblasUpper, CblasNoTrans, CblasNonUnit, m1, v1)
    for i in range(2):
        z = gsl_vector_complex_get(v1,i)
        ary.append(GSL_REAL(z) - Nv2[i].real)
        ary.append(GSL_IMAG(z) - Nv2[i].imag)

    # 2) v1 = m1^T * v1
    for i in range(2):
        z = gsl_complex_rect(Nv1[i].real, Nv1[i].imag)
        gsl_vector_complex_set(v1, i, z)
    Nv2 = dot(transpose(Nm1), Nv1)
    gsl_blas_ztrmv(CblasUpper, CblasTrans, CblasNonUnit, m1, v1)
    for i in range(2):
        z = gsl_vector_complex_get(v1,i)
        ary.append(GSL_REAL(z) - Nv2[i].real)
        ary.append(GSL_IMAG(z) - Nv2[i].imag)

    # 3) v1 = m1^H * v1
    for i in range(2):
        z = gsl_complex_rect(Nv1[i].real, Nv1[i].imag)
        gsl_vector_complex_set(v1, i, z)
    Nv2 = dot(conjugate(transpose(Nm1)), Nv1)
    gsl_blas_ztrmv(CblasUpper, CblasConjTrans, CblasNonUnit, m1, v1)
    for i in range(2):
        z = gsl_vector_complex_get(v1,i)
        ary.append(GSL_REAL(z) - Nv2[i].real)
        ary.append(GSL_IMAG(z) - Nv2[i].imag)

    # upper triangular matrix, with non-unit diagonal

    # 4) v1 = m1 * v1
    Nm1 = array([[1, 2+3j],[0, 1]])
    Nv2 = dot(Nm1, Nv1)
    for i in range(2):
        for j in range(2):
            z = gsl_complex_rect(Nm1[i,j].real, Nm1[i,j].imag)
            gsl_matrix_complex_set(m1, i, j, z)
    for i in range(2):
        z = gsl_complex_rect(Nv1[i].real, Nv1[i].imag)
        gsl_vector_complex_set(v1, i, z)
    gsl_blas_ztrmv(CblasUpper, CblasNoTrans, CblasUnit, m1, v1)
    for i in range(2):
        z = gsl_vector_complex_get(v1,i)
        ary.append(GSL_REAL(z) - Nv2[i].real)
        ary.append(GSL_IMAG(z) - Nv2[i].imag)

    # 5) v1 = m1^T * v1
    Nm1 = array([[1, 2+3j],[0, 1]])
    Nv2 = dot(transpose(Nm1), Nv1)
    for i in range(2):
        for j in range(2):
            z = gsl_complex_rect(Nm1[i,j].real, Nm1[i,j].imag)
            gsl_matrix_complex_set(m1, i, j, z)
    for i in range(2):
        z = gsl_complex_rect(Nv1[i].real, Nv1[i].imag)
        gsl_vector_complex_set(v1, i, z)
    gsl_blas_ztrmv(CblasUpper, CblasTrans, CblasUnit, m1, v1)
    for i in range(2):
        z = gsl_vector_complex_get(v1,i)
        ary.append(GSL_REAL(z) - Nv2[i].real)
        ary.append(GSL_IMAG(z) - Nv2[i].imag)

    # 6) v1 = m1^T * v1
    Nm1 = array([[1, 2+3j],[0, 1]])
    Nv2 = dot(conjugate(transpose(Nm1)), Nv1)
    for i in range(2):
        for j in range(2):
            z = gsl_complex_rect(Nm1[i,j].real, Nm1[i,j].imag)
            gsl_matrix_complex_set(m1, i, j, z)
    for i in range(2):
        z = gsl_complex_rect(Nv1[i].real, Nv1[i].imag)
        gsl_vector_complex_set(v1, i, z)
    gsl_blas_ztrmv(CblasUpper, CblasConjTrans, CblasUnit, m1, v1)
    for i in range(2):
        z = gsl_vector_complex_get(v1,i)
        ary.append(GSL_REAL(z) - Nv2[i].real)
        ary.append(GSL_IMAG(z) - Nv2[i].imag)

    gsl_matrix_complex_free(m1)
    gsl_vector_complex_free(v1)
    return ary

def t_gsl_blas_ztrsv():
    cdef gsl_complex z
    cdef gsl_matrix_complex * m1
    cdef gsl_vector_complex * v1
    m1 = gsl_matrix_complex_alloc(2,2)
    v1 =  gsl_vector_complex_alloc(2)
    ary = []
    '''
    CBLAS_UPLO_t = CblasUpper, CblasLower
    CBLAS_TRANSPOSE_t = CblasNoTrans, CblasTrans, CblasConjTrans
    CBLAS_DIAG_t = CblasNonUnit, CblasUpper
    There are 2*3*2 = 12 cases
    Here I check only the 6 cases with upper triangular matrix
    '''

    # upper triangular matrix, with non-unit diagonal
    Nm1 = array([[1+2j, 2+3j],[0, 6+7j]])
    Nv1 = array([7+1j, 3+2j])
    for i in range(2):
        for j in range(2):
            z = gsl_complex_rect(Nm1[i,j].real, Nm1[i,j].imag)
            gsl_matrix_complex_set(m1, i, j, z)
    for i in range(2):
        z = gsl_complex_rect(Nv1[i].real, Nv1[i].imag)
        gsl_vector_complex_set(v1, i, z)

    # 1) v1 = m1 * v1
    Nv2 = dot(inv(Nm1), Nv1)
    gsl_blas_ztrsv(CblasUpper, CblasNoTrans, CblasNonUnit, m1, v1)
    for i in range(2):
        z = gsl_vector_complex_get(v1,i)
        ary.append(GSL_REAL(z) - Nv2[i].real)
        ary.append(GSL_IMAG(z) - Nv2[i].imag)

    # 2) v1 = m1^T * v1
    for i in range(2):
        z = gsl_complex_rect(Nv1[i].real, Nv1[i].imag)
        gsl_vector_complex_set(v1, i, z)
    Nv2 = dot(inv(transpose(Nm1)), Nv1)
    gsl_blas_ztrsv(CblasUpper, CblasTrans, CblasNonUnit, m1, v1)
    for i in range(2):
        z = gsl_vector_complex_get(v1,i)
        ary.append(GSL_REAL(z) - Nv2[i].real)
        ary.append(GSL_IMAG(z) - Nv2[i].imag)

    # 3) v1 = m1^H * v1
    for i in range(2):
        z = gsl_complex_rect(Nv1[i].real, Nv1[i].imag)
        gsl_vector_complex_set(v1, i, z)
    Nv2 = dot(inv(conjugate(transpose(Nm1))), Nv1)
    gsl_blas_ztrsv(CblasUpper, CblasConjTrans, CblasNonUnit, m1, v1)
    for i in range(2):
        z = gsl_vector_complex_get(v1,i)
        ary.append(GSL_REAL(z) - Nv2[i].real)
        ary.append(GSL_IMAG(z) - Nv2[i].imag)

    # upper triangular matrix, with non-unit diagonal

    # 4) v1 = m1 * v1
    Nm1 = array([[1, 2+3j],[0, 1]])
    Nv2 = dot(inv(Nm1), Nv1)
    for i in range(2):
        for j in range(2):
            z = gsl_complex_rect(Nm1[i,j].real, Nm1[i,j].imag)
            gsl_matrix_complex_set(m1, i, j, z)
    for i in range(2):
        z = gsl_complex_rect(Nv1[i].real, Nv1[i].imag)
        gsl_vector_complex_set(v1, i, z)
    gsl_blas_ztrsv(CblasUpper, CblasNoTrans, CblasUnit, m1, v1)
    for i in range(2):
        z = gsl_vector_complex_get(v1,i)
        ary.append(GSL_REAL(z) - Nv2[i].real)
        ary.append(GSL_IMAG(z) - Nv2[i].imag)

    # 5) v1 = m1^T * v1
    Nm1 = array([[1, 2+3j],[0, 1]])
    Nv2 = dot(inv(transpose(Nm1)), Nv1)
    for i in range(2):
        for j in range(2):
            z = gsl_complex_rect(Nm1[i,j].real, Nm1[i,j].imag)
            gsl_matrix_complex_set(m1, i, j, z)
    for i in range(2):
        z = gsl_complex_rect(Nv1[i].real, Nv1[i].imag)
        gsl_vector_complex_set(v1, i, z)
    gsl_blas_ztrsv(CblasUpper, CblasTrans, CblasUnit, m1, v1)
    for i in range(2):
        z = gsl_vector_complex_get(v1,i)
        ary.append(GSL_REAL(z) - Nv2[i].real)
        ary.append(GSL_IMAG(z) - Nv2[i].imag)

    # 6) v1 = m1^T * v1
    Nm1 = array([[1, 2+3j],[0, 1]])
    Nv2 = dot(inv(conjugate(transpose(Nm1))), Nv1)
    for i in range(2):
        for j in range(2):
            z = gsl_complex_rect(Nm1[i,j].real, Nm1[i,j].imag)
            gsl_matrix_complex_set(m1, i, j, z)
    for i in range(2):
        z = gsl_complex_rect(Nv1[i].real, Nv1[i].imag)
        gsl_vector_complex_set(v1, i, z)
    gsl_blas_ztrsv(CblasUpper, CblasConjTrans, CblasUnit, m1, v1)
    for i in range(2):
        z = gsl_vector_complex_get(v1,i)
        ary.append(GSL_REAL(z) - Nv2[i].real)
        ary.append(GSL_IMAG(z) - Nv2[i].imag)

    gsl_matrix_complex_free(m1)
    gsl_vector_complex_free(v1)
    return ary

def t_gsl_blas_zhemv():
    cdef gsl_complex z, alpha, beta
    cdef gsl_matrix_complex * m1
    cdef gsl_vector_complex * v1, * v2
    m1 = gsl_matrix_complex_alloc(2,2)
    v1 =  gsl_vector_complex_alloc(2)
    v2 =  gsl_vector_complex_alloc(2)
    ary = []
    Nm1 = array([[1, 2+3j],[2-3j,2]])
    Nv1 = array([1+2j, 3 + 4j])
    Nv2 = array([1+3j, 2 + 4j])
    Nalpha = 0.1 + 0.2j
    Nbeta = 0.3 + 0.4j
    alpha = gsl_complex_rect(Nalpha.real, Nalpha.imag)
    beta = gsl_complex_rect(Nbeta.real, Nbeta.imag)
    Nv3 = Nalpha * dot(Nm1, Nv1) + Nbeta * Nv2

    for i in range(2):
        z = gsl_complex_rect(Nv1[i].real, Nv1[i].imag)
        gsl_vector_complex_set(v1, i, z)
        z = gsl_complex_rect(Nv2[i].real, Nv2[i].imag)
        gsl_vector_complex_set(v2, i, z)

    # CBLAS_UPLO_t = CblasUpper, CblasLower

    # upper matrix
    for i in range(2):
        for j in range(i,2):
            z = gsl_complex_rect(Nm1[i,j].real, Nm1[i,j].imag)
            gsl_matrix_complex_set(m1, i, j, z)
    gsl_blas_zhemv(CblasUpper, alpha, m1, v1, beta, v2)
    for i in range(0,2):
        z = gsl_vector_complex_get(v2, i)
        ary.append(GSL_REAL(z) - Nv3[i].real)
        ary.append(GSL_IMAG(z) - Nv3[i].imag)

    # lower matrix
    for i in range(2):
        z = gsl_complex_rect(Nv2[i].real, Nv2[i].imag)
        gsl_vector_complex_set(v2, i, z)
    for i in range(2):
        for j in range(0, i+1):
            z = gsl_complex_rect(Nm1[i,j].real, Nm1[i,j].imag)
            gsl_matrix_complex_set(m1, i, j, z)
    gsl_blas_zhemv(CblasLower, alpha, m1, v1, beta, v2)
    for i in range(0,2):
        z = gsl_vector_complex_get(v2, i)
        ary.append(GSL_REAL(z) - Nv3[i].real)
        ary.append(GSL_IMAG(z) - Nv3[i].imag)

    gsl_matrix_complex_free(m1)
    gsl_vector_complex_free(v1)
    gsl_vector_complex_free(v2)
    return ary

def t_gsl_blas_zgeru():
    cdef gsl_complex z, alpha
    cdef gsl_matrix_complex * m1
    cdef gsl_vector_complex * v1, * v2
    m1 = gsl_matrix_complex_alloc(2,2)
    v1 =  gsl_vector_complex_alloc(2)
    v2 =  gsl_vector_complex_alloc(2)
    ary = []
    Nm1 = array([[1, 2+3j],[2-3j,2]])
    Nv1 = array([1+2j, 3 + 4j])
    Nv2 = array([1+3j, 2 + 4j])
    Nalpha = 0.1 + 0.2j
    Nbeta = 0.3 + 0.4j
    Nm2 = Nalpha * outer(Nv1, Nv2) + Nm1

    alpha = gsl_complex_rect(Nalpha.real, Nalpha.imag)
    for i in range(2):
        z = gsl_complex_rect(Nv1[i].real, Nv1[i].imag)
        gsl_vector_complex_set(v1, i, z)
        z = gsl_complex_rect(Nv2[i].real, Nv2[i].imag)
        gsl_vector_complex_set(v2, i, z)

    for i in range(2):
        for j in range(0,2):
            z = gsl_complex_rect(Nm1[i,j].real, Nm1[i,j].imag)
            gsl_matrix_complex_set(m1, i, j, z)
    gsl_blas_zgeru(alpha, v1, v2, m1)
    for i in range(0,2):
        for j in range(0,2):
            z = gsl_matrix_complex_get(m1, i,j)
            ary.append(GSL_REAL(z) - Nm2[i,j].real)
            ary.append(GSL_IMAG(z) - Nm2[i,j].imag)

    gsl_matrix_complex_free(m1)
    gsl_vector_complex_free(v1)
    gsl_vector_complex_free(v2)
    return ary

def t_gsl_blas_zgerc():
    cdef gsl_complex z, alpha
    cdef gsl_matrix_complex * m1
    cdef gsl_vector_complex * v1, * v2
    m1 = gsl_matrix_complex_alloc(2,2)
    v1 =  gsl_vector_complex_alloc(2)
    v2 =  gsl_vector_complex_alloc(2)
    ary = []
    Nm1 = array([[1, 2+3j],[2-3j,2]])
    Nv1 = array([1+2j, 3 + 4j])
    Nv2 = array([1+3j, 2 + 4j])
    Nalpha = 0.1 + 0.2j
    Nbeta = 0.3 + 0.4j
    Nm2 = Nalpha * outer(Nv1, conjugate(Nv2)) + Nm1

    alpha = gsl_complex_rect(Nalpha.real, Nalpha.imag)
    for i in range(2):
        z = gsl_complex_rect(Nv1[i].real, Nv1[i].imag)
        gsl_vector_complex_set(v1, i, z)
        z = gsl_complex_rect(Nv2[i].real, Nv2[i].imag)
        gsl_vector_complex_set(v2, i, z)

    for i in range(2):
        for j in range(0,2):
            z = gsl_complex_rect(Nm1[i,j].real, Nm1[i,j].imag)
            gsl_matrix_complex_set(m1, i, j, z)
    gsl_blas_zgerc(alpha, v1, v2, m1)
    for i in range(0,2):
        for j in range(0,2):
            z = gsl_matrix_complex_get(m1, i,j)
            ary.append(GSL_REAL(z) - Nm2[i,j].real)
            ary.append(GSL_IMAG(z) - Nm2[i,j].imag)

    gsl_matrix_complex_free(m1)
    gsl_vector_complex_free(v1)
    gsl_vector_complex_free(v2)
    return ary

def t_gsl_blas_zher():
    cdef gsl_complex z
    cdef double alpha
    cdef gsl_matrix_complex * m1
    cdef gsl_vector_complex * v1
    m1 = gsl_matrix_complex_calloc(2,2)
    v1 =  gsl_vector_complex_alloc(2)
    ary = []
    Nm1 = array([[1, 2+3j], [2-3j,2]])
    Nv1 = array([1+2j, 3 + 4j])
    alpha = 0.1
    Nm2 = alpha * outer(Nv1, conjugate(Nv1)) + Nm1

    for i in range(2):
        z = gsl_complex_rect(Nv1[i].real, Nv1[i].imag)
        gsl_vector_complex_set(v1, i, z)

    for i in range(2):
        for j in range(i,2):
            z = gsl_complex_rect(Nm1[i,j].real, Nm1[i,j].imag)
            gsl_matrix_complex_set(m1, i, j, z)

    # 1) upper
    gsl_blas_zher(CblasUpper, alpha, v1, m1)
    for i in range(0,2):
        for j in range(i,2):
            z = gsl_matrix_complex_get(m1, i,j)
            ary.append(GSL_REAL(z) - Nm2[i,j].real)
            ary.append(GSL_IMAG(z) - Nm2[i,j].imag)

    # 2) lower
    for i in range(2):
        for j in range(0,2):
            z = gsl_complex_rect(Nm1[i,j].real, Nm1[i,j].imag)
            gsl_matrix_complex_set(m1, i, j, z)
    gsl_blas_zher(CblasLower, alpha, v1, m1)
    for i in range(0,2):
        for j in range(0, i+1):
            z = gsl_matrix_complex_get(m1, i,j)
            ary.append(GSL_REAL(z) - Nm2[i,j].real)
            ary.append(GSL_IMAG(z) - Nm2[i,j].imag)

    gsl_matrix_complex_free(m1)
    gsl_vector_complex_free(v1)
    return ary

def t_gsl_blas_zher2():
    cdef gsl_complex z, alpha
    cdef gsl_matrix_complex * m1
    cdef gsl_vector_complex * v1, * v2
    m1 = gsl_matrix_complex_calloc(2,2)
    v1 =  gsl_vector_complex_alloc(2)
    v2 =  gsl_vector_complex_alloc(2)
    Nalpha = 0.1 + 0.2j

    alpha = gsl_complex_rect(Nalpha.real, Nalpha.imag)
    ary = []
    Nm1 = array([[1, 2+3j], [2-3j,2]])
    Nv1 = array([1+2j, 3 + 4j])
    Nv2 = array([1+3j, 2 + 4j])
    Nm2 = Nalpha * outer(Nv1, conjugate(Nv2)) + \
          Nalpha.conjugate() * outer(Nv2, conjugate(Nv1)) + Nm1

    for i in range(2):
        z = gsl_complex_rect(Nv1[i].real, Nv1[i].imag)
        gsl_vector_complex_set(v1, i, z)
        z = gsl_complex_rect(Nv2[i].real, Nv2[i].imag)
        gsl_vector_complex_set(v2, i, z)

    for i in range(2):
        for j in range(i,2):
            z = gsl_complex_rect(Nm1[i,j].real, Nm1[i,j].imag)
            gsl_matrix_complex_set(m1, i, j, z)

    # 1) upper
    gsl_blas_zher2(CblasUpper, alpha, v1, v2, m1)
    for i in range(0,2):
        for j in range(i,2):
            z = gsl_matrix_complex_get(m1, i,j)
            ary.append(GSL_REAL(z) - Nm2[i,j].real)
            ary.append(GSL_IMAG(z) - Nm2[i,j].imag)

    # 2) lower
    for i in range(2):
        for j in range(0,2):
            z = gsl_complex_rect(Nm1[i,j].real, Nm1[i,j].imag)
            gsl_matrix_complex_set(m1, i, j, z)
    gsl_blas_zher2(CblasLower, alpha, v1, v2, m1)
    for i in range(0,2):
        for j in range(0, i+1):
            z = gsl_matrix_complex_get(m1, i,j)
            ary.append(GSL_REAL(z) - Nm2[i,j].real)
            ary.append(GSL_IMAG(z) - Nm2[i,j].imag)

    gsl_matrix_complex_free(m1)
    gsl_vector_complex_free(v1)
    return ary

def t_gsl_blas_dgemm():
    cdef gsl_matrix * m1, * m2, * m3
    cdef double alpha, beta
    m1 = gsl_matrix_alloc(2,2)
    m2 = gsl_matrix_alloc(2,2)
    m3 = gsl_matrix_alloc(2,2)
    alpha = 0.5
    beta = 0.6
    Nm1 = array([[1.1, 2], [3, 4]])
    Nm2 = array([[2.1, 3], [5, 4]])
    Nm3 = array([[3.1, 6], [5, 7]])
    ary = []
    for i in range(2):
        for j in range(2):
            gsl_matrix_set(m1,i,j, Nm1[i,j])
            gsl_matrix_set(m2,i,j, Nm2[i,j])
            gsl_matrix_set(m3,i,j, Nm3[i,j])

    '''
    CBLAS_TRANSPOSE_t = CblasNoTrans, CblasTrans
    CBLAS_TRANSPOSE_t = CblasNoTrans, CblasTrans
    There are 4 cases
    '''

    # 1)
    Nm4 = alpha * dot(Nm1, Nm2) + beta * Nm3

    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, alpha, m1, m2, beta, m3)
    for i in range(2):
        for j in range(2):
            ary.append(gsl_matrix_get(m3,i,j) - Nm4[i,j])

    # 2)
    for i in range(2):
        for j in range(2):
            gsl_matrix_set(m3,i,j, Nm3[i,j])
    Nm4 = alpha * dot(transpose(Nm1), Nm2) + beta * Nm3

    gsl_blas_dgemm(CblasTrans, CblasNoTrans, alpha, m1, m2, beta, m3)
    for i in range(2):
        for j in range(2):
            ary.append(gsl_matrix_get(m3,i,j) - Nm4[i,j])

    # 3)
    for i in range(2):
        for j in range(2):
            gsl_matrix_set(m3,i,j, Nm3[i,j])
    Nm4 = alpha * dot(Nm1, transpose(Nm2)) + beta * Nm3

    gsl_blas_dgemm(CblasNoTrans, CblasTrans, alpha, m1, m2, beta, m3)
    for i in range(2):
        for j in range(2):
            ary.append(gsl_matrix_get(m3,i,j) - Nm4[i,j])

    # 4)
    for i in range(2):
        for j in range(2):
            gsl_matrix_set(m3,i,j, Nm3[i,j])
    Nm4 = alpha * dot(transpose(Nm1), transpose(Nm2)) + beta * Nm3

    gsl_blas_dgemm(CblasTrans, CblasTrans, alpha, m1, m2, beta, m3)
    for i in range(2):
        for j in range(2):
            ary.append(gsl_matrix_get(m3,i,j) - Nm4[i,j])

    gsl_matrix_free(m1)
    gsl_matrix_free(m2)
    gsl_matrix_free(m3)
    return ary

def t_gsl_blas_dsymm():
    cdef gsl_matrix * m1, * m2, * m3
    cdef double alpha, beta
    m1 = gsl_matrix_calloc(2,2)
    m2 = gsl_matrix_alloc(2,2)
    m3 = gsl_matrix_alloc(2,2)
    alpha = 0.5
    beta = 0.6
    Nm1 = array([[1.1, 2], [2, 4]])
    Nm2 = array([[2.1, 3], [5, 4]])
    Nm3 = array([[3.1, 6], [5, 7]])
    ary = []
    for i in range(2):
        for j in range(2):
            gsl_matrix_set(m2,i,j, Nm2[i,j])
            gsl_matrix_set(m3,i,j, Nm3[i,j])

    '''
    CBLAS_SIDE_t = CblasLeft, CblasRight
    CBLAS_UPLO_t = CblasUpper, CblasLower
    There are 4 cases
    '''

    # 1)
    for i in range(2):
        for j in range(i,2):
            gsl_matrix_set(m1,i,j, Nm1[i,j])
    Nm4 = alpha * dot(Nm1, Nm2) + beta * Nm3
    gsl_blas_dsymm(CblasLeft, CblasUpper, alpha, m1, m2, beta, m3)
    for i in range(2):
        for j in range(2):
            ary.append(gsl_matrix_get(m3,i,j) - Nm4[i,j])

    # 2)
    for i in range(2):
        for j in range(2):
            gsl_matrix_set(m3,i,j, Nm3[i,j])
    Nm4 = alpha * dot(Nm2, Nm1) + beta * Nm3
    gsl_blas_dsymm(CblasRight, CblasUpper, alpha, m1, m2, beta, m3)
    for i in range(2):
        for j in range(2):
            ary.append(gsl_matrix_get(m3,i,j) - Nm4[i,j])

    # 3)
    gsl_matrix_set_zero(m1)
    for i in range(2):
        for j in range(0,i + 1):
            gsl_matrix_set(m1,i,j, Nm1[i,j])
    for i in range(2):
        for j in range(2):
            gsl_matrix_set(m3,i,j, Nm3[i,j])
    Nm4 = alpha * dot(Nm1, Nm2) + beta * Nm3
    gsl_blas_dsymm(CblasLeft, CblasLower, alpha, m1, m2, beta, m3)
    for i in range(2):
        for j in range(2):
            ary.append(gsl_matrix_get(m3,i,j) - Nm4[i,j])

    # 4)
    for i in range(2):
        for j in range(2):
            gsl_matrix_set(m3,i,j, Nm3[i,j])
    Nm4 = alpha * dot(Nm2, Nm1) + beta * Nm3
    gsl_blas_dsymm(CblasRight, CblasLower, alpha, m1, m2, beta, m3)
    for i in range(2):
        for j in range(2):
            ary.append(gsl_matrix_get(m3,i,j) - Nm4[i,j])

    gsl_matrix_free(m1)
    gsl_matrix_free(m2)
    gsl_matrix_free(m3)
    return ary

def t_gsl_blas_dtrmm():
    cdef gsl_matrix * m1, * m2
    cdef double alpha
    m1 = gsl_matrix_calloc(2,2)
    m2 = gsl_matrix_alloc(2,2)
    alpha = 0.5
    ary = []

    '''
    CBLAS_UPLO_t = CblasUpper, CblasLower
    CBLAS_DIAG_t = CblasUnit, CblasNonUnit
    CBLAS_SIDE_t = CblasLeft, CblasRight
    CBLAS_TRANSPOSE_t = CblasNoTrans, CblasTrans
    There are 16 cases
    '''

    # upper triangular non unit
    Nm1 = array([[1.1, 2], [0, 4]])
    Nm2 = array([[2.1, 3], [5, 4]])
    for i in range(2):
        for j in range(2):
            gsl_matrix_set(m1,i,j, Nm1[i,j])
            gsl_matrix_set(m2,i,j, Nm2[i,j])
    # 1) 1211
    Nm3 = alpha * dot(Nm1, Nm2)
    gsl_blas_dtrmm(CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, alpha, m1, m2)
    for i in range(2):
        for j in range(2):
            ary.append(gsl_matrix_get(m2,i,j) - Nm3[i,j])

    # 2) 1221
    for i in range(2):
        for j in range(2):
            gsl_matrix_set(m2,i,j, Nm2[i,j])
    Nm3 = alpha * dot(Nm2, Nm1)
    gsl_blas_dtrmm(CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit, alpha, m1, m2)
    for i in range(2):
        for j in range(2):
            ary.append(gsl_matrix_get(m2,i,j) - Nm3[i,j])

    # 3) 1212
    for i in range(2):
        for j in range(2):
            gsl_matrix_set(m2,i,j, Nm2[i,j])
    Nm3 = alpha * dot(transpose(Nm1), Nm2)
    gsl_blas_dtrmm(CblasLeft, CblasUpper, CblasTrans, CblasNonUnit, alpha, m1, m2)
    for i in range(2):
        for j in range(2):
            ary.append(gsl_matrix_get(m2,i,j) - Nm3[i,j])

    # 4) 1222
    for i in range(2):
        for j in range(2):
            gsl_matrix_set(m2,i,j, Nm2[i,j])
    Nm3 = alpha * dot(Nm2, transpose(Nm1))
    gsl_blas_dtrmm(CblasRight, CblasUpper, CblasTrans, CblasNonUnit, alpha, m1, m2)
    for i in range(2):
        for j in range(2):
            ary.append(gsl_matrix_get(m2,i,j) - Nm3[i,j])

    # upper triangular  unit
    Nm1 = array([[1, 2], [0, 1]])
    for i in range(2):
        for j in range(2):
            gsl_matrix_set(m1,i,j, Nm1[i,j])
    # 5) 1111
    for i in range(2):
        for j in range(2):
            gsl_matrix_set(m2,i,j, Nm2[i,j])
    Nm3 = alpha * dot(Nm1, Nm2)
    gsl_blas_dtrmm(CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, alpha, m1, m2)
    for i in range(2):
        for j in range(2):
            ary.append(gsl_matrix_get(m2,i,j) - Nm3[i,j])

    # 6) 1121
    for i in range(2):
        for j in range(2):
            gsl_matrix_set(m2,i,j, Nm2[i,j])
    Nm3 = alpha * dot(Nm2, Nm1)
    gsl_blas_dtrmm(CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit, alpha, m1, m2)
    for i in range(2):
        for j in range(2):
            ary.append(gsl_matrix_get(m2,i,j) - Nm3[i,j])

    # 7) 1112
    for i in range(2):
        for j in range(2):
            gsl_matrix_set(m2,i,j, Nm2[i,j])
    Nm3 = alpha * dot(transpose(Nm1), Nm2)
    gsl_blas_dtrmm(CblasLeft, CblasUpper, CblasTrans, CblasNonUnit, alpha, m1, m2)
    for i in range(2):
        for j in range(2):
            ary.append(gsl_matrix_get(m2,i,j) - Nm3[i,j])

    # 8) 1122
    for i in range(2):
        for j in range(2):
            gsl_matrix_set(m2,i,j, Nm2[i,j])
    Nm3 = alpha * dot(Nm2, transpose(Nm1))
    gsl_blas_dtrmm(CblasRight, CblasUpper, CblasTrans, CblasNonUnit, alpha, m1, m2)
    for i in range(2):
        for j in range(2):
            ary.append(gsl_matrix_get(m2,i,j) - Nm3[i,j])

    # lower triangular non unit
    Nm1 = array([[1.1, 0], [2, 4]])
    Nm2 = array([[2.1, 3], [5, 4]])
    for i in range(2):
        for j in range(2):
            gsl_matrix_set(m1,i,j, Nm1[i,j])
            gsl_matrix_set(m2,i,j, Nm2[i,j])
    # 9) 2211
    Nm3 = alpha * dot(Nm1, Nm2)
    gsl_blas_dtrmm(CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit, alpha, m1, m2)
    for i in range(2):
        for j in range(2):
            ary.append(gsl_matrix_get(m2,i,j) - Nm3[i,j])

    # 10) 2221
    for i in range(2):
        for j in range(2):
            gsl_matrix_set(m2,i,j, Nm2[i,j])
    Nm3 = alpha * dot(Nm2, Nm1)
    gsl_blas_dtrmm(CblasRight, CblasLower, CblasNoTrans, CblasNonUnit, alpha, m1, m2)
    for i in range(2):
        for j in range(2):
            ary.append(gsl_matrix_get(m2,i,j) - Nm3[i,j])

    # 11) 2212
    for i in range(2):
        for j in range(2):
            gsl_matrix_set(m2,i,j, Nm2[i,j])
    Nm3 = alpha * dot(transpose(Nm1), Nm2)
    gsl_blas_dtrmm(CblasLeft, CblasLower, CblasTrans, CblasNonUnit, alpha, m1, m2)
    for i in range(2):
        for j in range(2):
            ary.append(gsl_matrix_get(m2,i,j) - Nm3[i,j])

    # 12) 2222
    for i in range(2):
        for j in range(2):
            gsl_matrix_set(m2,i,j, Nm2[i,j])
    Nm3 = alpha * dot(Nm2, transpose(Nm1))
    gsl_blas_dtrmm(CblasRight, CblasLower, CblasTrans, CblasNonUnit, alpha, m1, m2)
    for i in range(2):
        for j in range(2):
            ary.append(gsl_matrix_get(m2,i,j) - Nm3[i,j])

    # lower triangular  unit
    Nm1 = array([[1, 0], [2, 1]])
    for i in range(2):
        for j in range(2):
            gsl_matrix_set(m1,i,j, Nm1[i,j])
    # 13) 2111
    for i in range(2):
        for j in range(2):
            gsl_matrix_set(m2,i,j, Nm2[i,j])
    Nm3 = alpha * dot(Nm1, Nm2)
    gsl_blas_dtrmm(CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit, alpha, m1, m2)
    for i in range(2):
        for j in range(2):
            ary.append(gsl_matrix_get(m2,i,j) - Nm3[i,j])

    # 14) 2121
    for i in range(2):
        for j in range(2):
            gsl_matrix_set(m2,i,j, Nm2[i,j])
    Nm3 = alpha * dot(Nm2, Nm1)
    gsl_blas_dtrmm(CblasRight, CblasLower, CblasNoTrans, CblasNonUnit, alpha, m1, m2)
    for i in range(2):
        for j in range(2):
            ary.append(gsl_matrix_get(m2,i,j) - Nm3[i,j])

    # 15) 2112
    for i in range(2):
        for j in range(2):
            gsl_matrix_set(m2,i,j, Nm2[i,j])
    Nm3 = alpha * dot(transpose(Nm1), Nm2)
    gsl_blas_dtrmm(CblasLeft, CblasLower, CblasTrans, CblasNonUnit, alpha, m1, m2)
    for i in range(2):
        for j in range(2):
            ary.append(gsl_matrix_get(m2,i,j) - Nm3[i,j])

    # 16) 2122
    for i in range(2):
        for j in range(2):
            gsl_matrix_set(m2,i,j, Nm2[i,j])
    Nm3 = alpha * dot(Nm2, transpose(Nm1))
    gsl_blas_dtrmm(CblasRight, CblasLower, CblasTrans, CblasNonUnit, alpha, m1, m2)
    for i in range(2):
        for j in range(2):
            ary.append(gsl_matrix_get(m2,i,j) - Nm3[i,j])

    gsl_matrix_free(m1)
    gsl_matrix_free(m2)
    return ary

def t_gsl_blas_dtrsm():
    cdef gsl_matrix * m1, * m2
    cdef double alpha
    m1 = gsl_matrix_calloc(2,2)
    m2 = gsl_matrix_alloc(2,2)
    alpha = 0.5
    ary = []

    '''
    CBLAS_UPLO_t = CblasUpper, CblasLower
    CBLAS_DIAG_t = CblasUnit, CblasNonUnit
    CBLAS_SIDE_t = CblasLeft, CblasRight
    CBLAS_TRANSPOSE_t = CblasNoTrans, CblasTrans
    There are 16 cases
    '''

    # upper triangular non unit
    Nm1 = array([[1.1, 2], [0, 4]])
    Nm2 = array([[2.1, 3], [5, 4]])
    for i in range(2):
        for j in range(2):
            gsl_matrix_set(m1,i,j, Nm1[i,j])
            gsl_matrix_set(m2,i,j, Nm2[i,j])
    # 1) 1211
    Nm3 = alpha * dot(inv(Nm1), Nm2)
    gsl_blas_dtrsm(CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, alpha, m1, m2)
    for i in range(2):
        for j in range(2):
            ary.append(gsl_matrix_get(m2,i,j) - Nm3[i,j])

    # 2) 1221
    for i in range(2):
        for j in range(2):
            gsl_matrix_set(m2,i,j, Nm2[i,j])
    Nm3 = alpha * dot(Nm2, inv(Nm1))
    gsl_blas_dtrsm(CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit, alpha, m1, m2)
    for i in range(2):
        for j in range(2):
            ary.append(gsl_matrix_get(m2,i,j) - Nm3[i,j])

    # 3) 1212
    for i in range(2):
        for j in range(2):
            gsl_matrix_set(m2,i,j, Nm2[i,j])
    Nm3 = alpha * dot(transpose(inv(Nm1)), Nm2)
    gsl_blas_dtrsm(CblasLeft, CblasUpper, CblasTrans, CblasNonUnit, alpha, m1, m2)
    for i in range(2):
        for j in range(2):
            ary.append(gsl_matrix_get(m2,i,j) - Nm3[i,j])

    # 4) 1222
    for i in range(2):
        for j in range(2):
            gsl_matrix_set(m2,i,j, Nm2[i,j])
    Nm3 = alpha * dot(Nm2, transpose(inv(Nm1)))
    gsl_blas_dtrsm(CblasRight, CblasUpper, CblasTrans, CblasNonUnit, alpha, m1, m2)
    for i in range(2):
        for j in range(2):
            ary.append(gsl_matrix_get(m2,i,j) - Nm3[i,j])

    # upper triangular  unit
    Nm1 = array([[1, 2], [0, 1]])
    for i in range(2):
        for j in range(2):
            gsl_matrix_set(m1,i,j, Nm1[i,j])
    # 5) 1111
    for i in range(2):
        for j in range(2):
            gsl_matrix_set(m2,i,j, Nm2[i,j])
    Nm3 = alpha * dot(inv(Nm1), Nm2)
    gsl_blas_dtrsm(CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, alpha, m1, m2)
    for i in range(2):
        for j in range(2):
            ary.append(gsl_matrix_get(m2,i,j) - Nm3[i,j])

    # 6) 1121
    for i in range(2):
        for j in range(2):
            gsl_matrix_set(m2,i,j, Nm2[i,j])
    Nm3 = alpha * dot(Nm2, inv(Nm1))
    gsl_blas_dtrsm(CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit, alpha, m1, m2)
    for i in range(2):
        for j in range(2):
            ary.append(gsl_matrix_get(m2,i,j) - Nm3[i,j])

    # 7) 1112
    for i in range(2):
        for j in range(2):
            gsl_matrix_set(m2,i,j, Nm2[i,j])
    Nm3 = alpha * dot(transpose(inv(Nm1)), Nm2)
    gsl_blas_dtrsm(CblasLeft, CblasUpper, CblasTrans, CblasNonUnit, alpha, m1, m2)
    for i in range(2):
        for j in range(2):
            ary.append(gsl_matrix_get(m2,i,j) - Nm3[i,j])

    # 8) 1122
    for i in range(2):
        for j in range(2):
            gsl_matrix_set(m2,i,j, Nm2[i,j])
    Nm3 = alpha * dot(Nm2, transpose(inv(Nm1)))
    gsl_blas_dtrsm(CblasRight, CblasUpper, CblasTrans, CblasNonUnit, alpha, m1, m2)
    for i in range(2):
        for j in range(2):
            ary.append(gsl_matrix_get(m2,i,j) - Nm3[i,j])

    # lower triangular non unit
    Nm1 = array([[1.1, 0], [2, 4]])
    Nm2 = array([[2.1, 3], [5, 4]])
    for i in range(2):
        for j in range(2):
            gsl_matrix_set(m1,i,j, Nm1[i,j])
            gsl_matrix_set(m2,i,j, Nm2[i,j])
    # 9) 2211
    Nm3 = alpha * dot(inv(Nm1), Nm2)
    gsl_blas_dtrsm(CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit, alpha, m1, m2)
    for i in range(2):
        for j in range(2):
            ary.append(gsl_matrix_get(m2,i,j) - Nm3[i,j])

    # 10) 2221
    for i in range(2):
        for j in range(2):
            gsl_matrix_set(m2,i,j, Nm2[i,j])
    Nm3 = alpha * dot(Nm2, inv(Nm1))
    gsl_blas_dtrsm(CblasRight, CblasLower, CblasNoTrans, CblasNonUnit, alpha, m1, m2)
    for i in range(2):
        for j in range(2):
            ary.append(gsl_matrix_get(m2,i,j) - Nm3[i,j])

    # 11) 2212
    for i in range(2):
        for j in range(2):
            gsl_matrix_set(m2,i,j, Nm2[i,j])
    Nm3 = alpha * dot(transpose(inv(Nm1)), Nm2)
    gsl_blas_dtrsm(CblasLeft, CblasLower, CblasTrans, CblasNonUnit, alpha, m1, m2)
    for i in range(2):
        for j in range(2):
            ary.append(gsl_matrix_get(m2,i,j) - Nm3[i,j])

    # 12) 2222
    for i in range(2):
        for j in range(2):
            gsl_matrix_set(m2,i,j, Nm2[i,j])
    Nm3 = alpha * dot(Nm2, transpose(inv(Nm1)))
    gsl_blas_dtrsm(CblasRight, CblasLower, CblasTrans, CblasNonUnit, alpha, m1, m2)
    for i in range(2):
        for j in range(2):
            ary.append(gsl_matrix_get(m2,i,j) - Nm3[i,j])

    # lower triangular  unit
    Nm1 = array([[1, 0], [2, 1]])
    for i in range(2):
        for j in range(2):
            gsl_matrix_set(m1,i,j, Nm1[i,j])
    # 13) 2111
    for i in range(2):
        for j in range(2):
            gsl_matrix_set(m2,i,j, Nm2[i,j])
    Nm3 = alpha * dot(inv(Nm1), Nm2)
    gsl_blas_dtrsm(CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit, alpha, m1, m2)
    for i in range(2):
        for j in range(2):
            ary.append(gsl_matrix_get(m2,i,j) - Nm3[i,j])

    # 14) 2121
    for i in range(2):
        for j in range(2):
            gsl_matrix_set(m2,i,j, Nm2[i,j])
    Nm3 = alpha * dot(Nm2, inv(Nm1))
    gsl_blas_dtrsm(CblasRight, CblasLower, CblasNoTrans, CblasNonUnit, alpha, m1, m2)
    for i in range(2):
        for j in range(2):
            ary.append(gsl_matrix_get(m2,i,j) - Nm3[i,j])

    # 15) 2112
    for i in range(2):
        for j in range(2):
            gsl_matrix_set(m2,i,j, Nm2[i,j])
    Nm3 = alpha * dot(transpose(inv(Nm1)), Nm2)
    gsl_blas_dtrsm(CblasLeft, CblasLower, CblasTrans, CblasNonUnit, alpha, m1, m2)
    for i in range(2):
        for j in range(2):
            ary.append(gsl_matrix_get(m2,i,j) - Nm3[i,j])

    # 16) 2122
    for i in range(2):
        for j in range(2):
            gsl_matrix_set(m2,i,j, Nm2[i,j])
    Nm3 = alpha * dot(Nm2, transpose(inv(Nm1)))
    gsl_blas_dtrsm(CblasRight, CblasLower, CblasTrans, CblasNonUnit, alpha, m1, m2)
    for i in range(2):
        for j in range(2):
            ary.append(gsl_matrix_get(m2,i,j) - Nm3[i,j])

    gsl_matrix_free(m1)
    gsl_matrix_free(m2)
    return ary


def t_gsl_blas_dsyrk():
    cdef gsl_matrix * m1, * m2
    cdef double alpha, beta
    m1 = gsl_matrix_alloc(2,2)
    m2 = gsl_matrix_calloc(2,2)
    alpha = 0.5
    beta = 0.6
    ary = []

    '''
    CBLAS_UPLO_t = CblasUpper, CblasLower
    CBLAS_TRANSPOSE_t = CblasNoTrans, CblasTrans
    There are 4 cases
    '''

    # upper triangular part of m2 used
    Nm1 = array([[1.1, 2], [3, 4]])
    Nm2 = array([[2.1, 3], [3, 4]])
    for i in range(2):
        for j in range(2):
            gsl_matrix_set(m1,i,j, Nm1[i,j])
    for i in range(2):
        for j in range(i,2):
            gsl_matrix_set(m2,i,j, Nm2[i,j])
    # 1)
    Nm3 = alpha * dot(Nm1, transpose(Nm1)) + beta * Nm2
    gsl_blas_dsyrk(CblasUpper, CblasNoTrans, alpha, m1, beta, m2)
    for i in range(2):
        for j in range(i, 2):
            ary.append(gsl_matrix_get(m2,i,j) - Nm3[i,j])

    # 2)
    for i in range(2):
        for j in range(i,2):
            gsl_matrix_set(m2,i,j, Nm2[i,j])
    Nm3 = alpha * dot(transpose(Nm1), Nm1) + beta * Nm2
    gsl_blas_dsyrk(CblasUpper, CblasTrans, alpha, m1, beta, m2)
    for i in range(2):
        for j in range(i, 2):
            ary.append(gsl_matrix_get(m2,i,j) - Nm3[i,j])

    # lower triangular part of m2 used
    gsl_matrix_set_zero(m2)
    for i in range(2):
        for j in range(0, i+1):
            gsl_matrix_set(m2,i,j, Nm2[i,j])

    # 3)
    Nm3 = alpha * dot(Nm1, transpose(Nm1)) + beta * Nm2
    gsl_blas_dsyrk(CblasLower, CblasNoTrans, alpha, m1, beta, m2)
    for i in range(2):
        for j in range(0, i+1):
            ary.append(gsl_matrix_get(m2,i,j) - Nm3[i,j])

    # 4)
    for i in range(2):
        for j in range(0, i+1):
            gsl_matrix_set(m2,i,j, Nm2[i,j])
    Nm3 = alpha * dot(transpose(Nm1), Nm1) + beta * Nm2
    gsl_blas_dsyrk(CblasLower, CblasTrans, alpha, m1, beta, m2)
    for i in range(2):
        for j in range(0, i+1):
            ary.append(gsl_matrix_get(m2,i,j) - Nm3[i,j])

    gsl_matrix_free(m1)
    gsl_matrix_free(m2)
    return ary

def t_gsl_blas_dsyr2k():
    cdef gsl_matrix * m1, * m2, * m3
    cdef double alpha, beta
    m1 = gsl_matrix_alloc(2,2)
    m2 = gsl_matrix_alloc(2,2)
    m3 = gsl_matrix_calloc(2,2)
    alpha = 0.5
    beta = 0.6
    ary = []

    '''
    CBLAS_UPLO_t = CblasUpper, CblasLower
    CBLAS_TRANSPOSE_t = CblasNoTrans, CblasTrans
    There are 4 cases
    '''

    # upper triangular part of m3 used
    Nm1 = array([[1.1, 2], [3, 4]])
    Nm2 = array([[1.2, 5], [6, 4]])
    Nm3 = array([[2.1, 3], [3, 4]])
    for i in range(2):
        for j in range(2):
            gsl_matrix_set(m1,i,j, Nm1[i,j])
            gsl_matrix_set(m2,i,j, Nm2[i,j])
    for i in range(2):
        for j in range(i,2):
            gsl_matrix_set(m3,i,j, Nm3[i,j])
    # 1)
    Nm4 = alpha * dot(Nm1, transpose(Nm2)) + \
          alpha * dot(Nm2, transpose(Nm1)) + beta * Nm3
    gsl_blas_dsyr2k(CblasUpper, CblasNoTrans, alpha, m1, m2, beta, m3)
    for i in range(2):
        for j in range(i, 2):
            ary.append(gsl_matrix_get(m3,i,j) - Nm4[i,j])

    # 2)
    for i in range(2):
        for j in range(i,2):
            gsl_matrix_set(m3,i,j, Nm3[i,j])
    Nm4 = alpha * dot(transpose(Nm1), Nm2) + \
          alpha * dot(transpose(Nm2), Nm1) + beta * Nm3
    gsl_blas_dsyr2k(CblasUpper, CblasTrans, alpha, m1, m2, beta, m3)
    for i in range(2):
        for j in range(i, 2):
            ary.append(gsl_matrix_get(m3,i,j) - Nm4[i,j])

    # lower triangular part of m3 used
    gsl_matrix_set_zero(m3)
    for i in range(2):
        for j in range(0, i+1):
            gsl_matrix_set(m3,i,j, Nm3[i,j])

    # 3)
    gsl_matrix_set_zero(m3)
    for i in range(2):
        for j in range(0, i+1):
            gsl_matrix_set(m3,i,j, Nm3[i,j])
    Nm4 = alpha * dot(Nm1, transpose(Nm2)) + \
          alpha * dot(Nm2, transpose(Nm1)) + beta * Nm3
    gsl_blas_dsyr2k(CblasLower, CblasNoTrans, alpha, m1, m2, beta, m3)
    for i in range(2):
        for j in range(0, i+1):
            ary.append(gsl_matrix_get(m3,i,j) - Nm4[i,j])

    # 4)
    for i in range(2):
        for j in range(0, i+1):
            gsl_matrix_set(m3,i,j, Nm3[i,j])
    Nm4 = alpha * dot(transpose(Nm1), Nm2) + \
          alpha * dot(transpose(Nm2), Nm1) + beta * Nm3
    gsl_blas_dsyr2k(CblasLower, CblasTrans, alpha, m1, m2, beta, m3)
    for i in range(2):
        for j in range(0, i+1):
            ary.append(gsl_matrix_get(m3,i,j) - Nm4[i,j])

    gsl_matrix_free(m1)
    gsl_matrix_free(m2)
    return ary


def t_gsl_blas_zgemm():
    cdef gsl_matrix_complex * m1, * m2, * m3
    cdef gsl_complex z, alpha, beta
    m1 = gsl_matrix_complex_alloc(2,2)
    m2 = gsl_matrix_complex_alloc(2,2)
    m3 = gsl_matrix_complex_alloc(2,2)
    Nalpha = 0.1 + 0.2j
    Nbeta = 0.3 + 0.4j
    Nm1 = array([[1+2j,3+4j],[4+5j,5+6j]])
    Nm2 = array([[3+2j,3+5j],[3+5j,7+6j]])
    Nm3 = array([[1+1j,3+5j],[7+5j,5+8j]])
    alpha = gsl_complex_rect(Nalpha.real, Nalpha.imag)
    beta = gsl_complex_rect(Nbeta.real, Nbeta.imag)
    ary = []
    for i in range(2):
        for j in range(2):
            z = gsl_complex_rect(Nm1[i,j].real, Nm1[i,j].imag)
            gsl_matrix_complex_set(m1, i, j, z)
            z = gsl_complex_rect(Nm2[i,j].real, Nm2[i,j].imag)
            gsl_matrix_complex_set(m2, i, j, z)
            z = gsl_complex_rect(Nm3[i,j].real, Nm3[i,j].imag)
            gsl_matrix_complex_set(m3, i, j, z)

    # CBLAS_TRANSPOSE_t = CblasNoTrans, CblasTrans, CblasConjTrans
    # CBLAS_TRANSPOSE_t = CblasNoTrans, CblasTrans, CblasConjTrans
    # 9 cases

    # 1) 11
    Nm4 = Nalpha * dot(Nm1,Nm2) + Nbeta * Nm3
    gsl_blas_zgemm(CblasNoTrans,CblasNoTrans,alpha, m1, m2, beta, m3)
    for i in range(2):
        for j in range(2):
            z = gsl_matrix_complex_get(m3,i,j)
            ary.append(GSL_REAL(z) - Nm4[i,j].real)
            ary.append(GSL_IMAG(z) - Nm4[i,j].imag)

    # 2) 12
    for i in range(2):
        for j in range(2):
            z = gsl_complex_rect(Nm3[i,j].real, Nm3[i,j].imag)
            gsl_matrix_complex_set(m3, i, j, z)
    Nm4 = Nalpha * dot(Nm1,transpose(Nm2)) + Nbeta * Nm3
    gsl_blas_zgemm(CblasNoTrans,CblasTrans,alpha, m1, m2, beta, m3)
    for i in range(2):
        for j in range(2):
            z = gsl_matrix_complex_get(m3,i,j)
            ary.append(GSL_REAL(z) - Nm4[i,j].real)
            ary.append(GSL_IMAG(z) - Nm4[i,j].imag)

    # 3) 21
    for i in range(2):
        for j in range(2):
            z = gsl_complex_rect(Nm3[i,j].real, Nm3[i,j].imag)
            gsl_matrix_complex_set(m3, i, j, z)
    Nm4 = Nalpha * dot(transpose(Nm1), Nm2) + Nbeta * Nm3
    gsl_blas_zgemm(CblasTrans,CblasNoTrans,alpha, m1, m2, beta, m3)
    for i in range(2):
        for j in range(2):
            z = gsl_matrix_complex_get(m3,i,j)
            ary.append(GSL_REAL(z) - Nm4[i,j].real)
            ary.append(GSL_IMAG(z) - Nm4[i,j].imag)

    # 4) 22
    for i in range(2):
        for j in range(2):
            z = gsl_complex_rect(Nm3[i,j].real, Nm3[i,j].imag)
            gsl_matrix_complex_set(m3, i, j, z)
    Nm4 = Nalpha * dot(transpose(Nm1),transpose(Nm2)) + Nbeta * Nm3
    gsl_blas_zgemm(CblasTrans,CblasTrans,alpha, m1, m2, beta, m3)
    for i in range(2):
        for j in range(2):
            z = gsl_matrix_complex_get(m3,i,j)
            ary.append(GSL_REAL(z) - Nm4[i,j].real)
            ary.append(GSL_IMAG(z) - Nm4[i,j].imag)

    # 5) 13
    for i in range(2):
        for j in range(2):
            z = gsl_complex_rect(Nm3[i,j].real, Nm3[i,j].imag)
            gsl_matrix_complex_set(m3, i, j, z)
    Nm4 = Nalpha * dot(Nm1,conjugate(transpose(Nm2))) + Nbeta * Nm3
    gsl_blas_zgemm(CblasNoTrans,CblasConjTrans,alpha, m1, m2, beta, m3)
    for i in range(2):
        for j in range(2):
            z = gsl_matrix_complex_get(m3,i,j)
            ary.append(GSL_REAL(z) - Nm4[i,j].real)
            ary.append(GSL_IMAG(z) - Nm4[i,j].imag)

    # 6) 31
    for i in range(2):
        for j in range(2):
            z = gsl_complex_rect(Nm3[i,j].real, Nm3[i,j].imag)
            gsl_matrix_complex_set(m3, i, j, z)
    Nm4 = Nalpha * dot(conjugate(transpose(Nm1)), Nm2) + Nbeta * Nm3
    gsl_blas_zgemm(CblasConjTrans,CblasNoTrans,alpha, m1, m2, beta, m3)
    for i in range(2):
        for j in range(2):
            z = gsl_matrix_complex_get(m3,i,j)
            ary.append(GSL_REAL(z) - Nm4[i,j].real)
            ary.append(GSL_IMAG(z) - Nm4[i,j].imag)

    # 7) 33
    for i in range(2):
        for j in range(2):
            z = gsl_complex_rect(Nm3[i,j].real, Nm3[i,j].imag)
            gsl_matrix_complex_set(m3, i, j, z)
    Nm4 = Nalpha * dot(conjugate(transpose(Nm1)),conjugate(transpose(Nm2))) + Nbeta * Nm3
    gsl_blas_zgemm(CblasConjTrans,CblasConjTrans,alpha, m1, m2, beta, m3)
    for i in range(2):
        for j in range(2):
            z = gsl_matrix_complex_get(m3,i,j)
            ary.append(GSL_REAL(z) - Nm4[i,j].real)
            ary.append(GSL_IMAG(z) - Nm4[i,j].imag)

    # 8) 23
    for i in range(2):
        for j in range(2):
            z = gsl_complex_rect(Nm3[i,j].real, Nm3[i,j].imag)
            gsl_matrix_complex_set(m3, i, j, z)
    Nm4 = Nalpha * dot(transpose(Nm1),conjugate(transpose(Nm2))) + Nbeta * Nm3
    gsl_blas_zgemm(CblasTrans,CblasConjTrans,alpha, m1, m2, beta, m3)
    for i in range(2):
        for j in range(2):
            z = gsl_matrix_complex_get(m3,i,j)
            ary.append(GSL_REAL(z) - Nm4[i,j].real)
            ary.append(GSL_IMAG(z) - Nm4[i,j].imag)

    # 9) 32
    for i in range(2):
        for j in range(2):
            z = gsl_complex_rect(Nm3[i,j].real, Nm3[i,j].imag)
            gsl_matrix_complex_set(m3, i, j, z)
    Nm4 = Nalpha * dot(conjugate(transpose(Nm1)), transpose(Nm2)) + Nbeta * Nm3
    gsl_blas_zgemm(CblasConjTrans,CblasTrans,alpha, m1, m2, beta, m3)
    for i in range(2):
        for j in range(2):
            z = gsl_matrix_complex_get(m3,i,j)
            ary.append(GSL_REAL(z) - Nm4[i,j].real)
            ary.append(GSL_IMAG(z) - Nm4[i,j].imag)

    gsl_matrix_complex_free(m1)
    gsl_matrix_complex_free(m2)
    gsl_matrix_complex_free(m3)
    return ary

def t_gsl_blas_zsymm():
    cdef gsl_matrix_complex * m1, * m2, * m3
    cdef gsl_complex z, alpha, beta
    m1 = gsl_matrix_complex_alloc(2,2)
    m2 = gsl_matrix_complex_alloc(2,2)
    m3 = gsl_matrix_complex_alloc(2,2)
    Nalpha = 0.1 + 0.2j
    Nbeta = 0.3 + 0.4j
    Nm1 = array([[1+2j,3+4j],[3+4j,5+6j]])
    Nm2 = array([[3+2j,3+5j],[3+5j,7+6j]])
    Nm3 = array([[1+1j,3+5j],[7+5j,5+8j]])
    alpha = gsl_complex_rect(Nalpha.real, Nalpha.imag)
    beta = gsl_complex_rect(Nbeta.real, Nbeta.imag)
    ary = []
    for i in range(2):
        for j in range(2):
            z = gsl_complex_rect(Nm2[i,j].real, Nm2[i,j].imag)
            gsl_matrix_complex_set(m2, i, j, z)
            z = gsl_complex_rect(Nm3[i,j].real, Nm3[i,j].imag)
            gsl_matrix_complex_set(m3, i, j, z)

    '''
    CBLAS_UPLO_t = CblasUpper, CblasLower
    CBLAS_SIDE_t = CblasLeft, CblasRight
    There are 4 cases
    '''

    # 1) 11
    for i in range(2):
        for j in range(i,2):
            z = gsl_complex_rect(Nm1[i,j].real, Nm1[i,j].imag)
            gsl_matrix_complex_set(m1, i, j, z)
    Nm4 = Nalpha * dot(Nm1,Nm2) + Nbeta * Nm3
    gsl_blas_zsymm(CblasLeft, CblasUpper,alpha, m1, m2, beta, m3)
    for i in range(2):
        for j in range(2):
            z = gsl_matrix_complex_get(m3,i,j)
            ary.append(GSL_REAL(z) - Nm4[i,j].real)
            ary.append(GSL_IMAG(z) - Nm4[i,j].imag)

    # 2) 12
    for i in range(2):
        for j in range(0,2):
            z = gsl_complex_rect(Nm3[i,j].real, Nm3[i,j].imag)
            gsl_matrix_complex_set(m3, i, j, z)
    Nm4 = Nalpha * dot(Nm2,Nm1) + Nbeta * Nm3
    gsl_blas_zsymm(CblasRight, CblasUpper,alpha, m1, m2, beta, m3)
    for i in range(2):
        for j in range(2):
            z = gsl_matrix_complex_get(m3,i,j)
            ary.append(GSL_REAL(z) - Nm4[i,j].real)
            ary.append(GSL_IMAG(z) - Nm4[i,j].imag)

    # 3) 21
    gsl_matrix_complex_set_zero(m1)
    for i in range(2):
        for j in range(0,i+1):
            z = gsl_complex_rect(Nm1[i,j].real, Nm1[i,j].imag)
            gsl_matrix_complex_set(m1, i, j, z)
    for i in range(2):
        for j in range(0,2):
            z = gsl_complex_rect(Nm3[i,j].real, Nm3[i,j].imag)
            gsl_matrix_complex_set(m3, i, j, z)
    Nm4 = Nalpha * dot(Nm1,Nm2) + Nbeta * Nm3
    gsl_blas_zsymm(CblasLeft, CblasLower,alpha, m1, m2, beta, m3)
    for i in range(2):
        for j in range(2):
            z = gsl_matrix_complex_get(m3,i,j)
            ary.append(GSL_REAL(z) - Nm4[i,j].real)
            ary.append(GSL_IMAG(z) - Nm4[i,j].imag)

    # 4) 22
    for i in range(2):
        for j in range(0,2):
            z = gsl_complex_rect(Nm3[i,j].real, Nm3[i,j].imag)
            gsl_matrix_complex_set(m3, i, j, z)
    Nm4 = Nalpha * dot(Nm2,Nm1) + Nbeta * Nm3
    gsl_blas_zsymm(CblasRight, CblasLower,alpha, m1, m2, beta, m3)
    for i in range(2):
        for j in range(2):
            z = gsl_matrix_complex_get(m3,i,j)
            ary.append(GSL_REAL(z) - Nm4[i,j].real)
            ary.append(GSL_IMAG(z) - Nm4[i,j].imag)


    gsl_matrix_complex_free(m1)
    gsl_matrix_complex_free(m2)
    gsl_matrix_complex_free(m3)
    return ary

def t_gsl_blas_zsyrk():
    cdef gsl_matrix_complex * m1, * m2
    cdef gsl_complex z, alpha, beta
    m1 = gsl_matrix_complex_alloc(2,2)
    m2 = gsl_matrix_complex_alloc(2,2)
    Nalpha = 0.1 + 0.2j
    Nbeta = 0.3 + 0.4j
    Nm1 = array([[1+2j,3+4j],[5+4j,5+6j]])
    Nm2 = array([[3+2j,3+5j],[3+5j,2+6j]])
    alpha = gsl_complex_rect(Nalpha.real, Nalpha.imag)
    beta = gsl_complex_rect(Nbeta.real, Nbeta.imag)
    ary = []
    for i in range(2):
        for j in range(2):
            z = gsl_complex_rect(Nm1[i,j].real, Nm1[i,j].imag)
            gsl_matrix_complex_set(m1, i, j, z)

    '''
    CBLAS_UPLO_t = CblasUpper, CblasLower
    CBLAS_TRANSPOSE_t = CblasNoTrans, CblasTrans
    There are 4 cases
    '''

    # upper triangular part of m2 used
    # 1) 11
    for i in range(2):
        for j in range(i,2):
            z = gsl_complex_rect(Nm2[i,j].real, Nm2[i,j].imag)
            gsl_matrix_complex_set(m2, i, j, z)
    Nm3 = Nalpha * dot(Nm1,transpose(Nm1)) + Nbeta * Nm2
    gsl_blas_zsyrk(CblasUpper, CblasNoTrans,alpha, m1, beta, m2)
    for i in range(2):
        for j in range(i,2):
            z = gsl_matrix_complex_get(m2,i,j)
            ary.append(GSL_REAL(z) - Nm3[i,j].real)
            ary.append(GSL_IMAG(z) - Nm3[i,j].imag)

    # 2) 12
    for i in range(2):
        for j in range(i,2):
            z = gsl_complex_rect(Nm2[i,j].real, Nm2[i,j].imag)
            gsl_matrix_complex_set(m2, i, j, z)
    Nm3 = Nalpha * dot(transpose(Nm1), Nm1) + Nbeta * Nm2
    gsl_blas_zsyrk(CblasUpper, CblasTrans,alpha, m1, beta, m2)
    for i in range(2):
        for j in range(i,2):
            z = gsl_matrix_complex_get(m2,i,j)
            ary.append(GSL_REAL(z) - Nm3[i,j].real)
            ary.append(GSL_IMAG(z) - Nm3[i,j].imag)

    # lower triangular part of m2 used
    # 3) 21
    for i in range(2):
        for j in range(0,i+1):
            z = gsl_complex_rect(Nm2[i,j].real, Nm2[i,j].imag)
            gsl_matrix_complex_set(m2, i, j, z)
    Nm3 = Nalpha * dot(Nm1,transpose(Nm1)) + Nbeta * Nm2
    gsl_blas_zsyrk(CblasLower, CblasNoTrans,alpha, m1, beta, m2)
    for i in range(2):
        for j in range(0,i+1):
            z = gsl_matrix_complex_get(m2,i,j)
            ary.append(GSL_REAL(z) - Nm3[i,j].real)
            ary.append(GSL_IMAG(z) - Nm3[i,j].imag)

    # 4) 22
    for i in range(2):
        for j in range(0,i+1):
            z = gsl_complex_rect(Nm2[i,j].real, Nm2[i,j].imag)
            gsl_matrix_complex_set(m2, i, j, z)
    Nm3 = Nalpha * dot(transpose(Nm1),Nm1) + Nbeta * Nm2
    gsl_blas_zsyrk(CblasLower, CblasTrans,alpha, m1, beta, m2)
    for i in range(2):
        for j in range(0,i+1):
            z = gsl_matrix_complex_get(m2,i,j)
            ary.append(GSL_REAL(z) - Nm3[i,j].real)
            ary.append(GSL_IMAG(z) - Nm3[i,j].imag)

    gsl_matrix_complex_free(m1)
    gsl_matrix_complex_free(m2)
    return ary

def t_gsl_blas_zsyr2k():
    cdef gsl_matrix_complex * m1, * m2, * m3
    cdef gsl_complex z, alpha, beta
    m1 = gsl_matrix_complex_alloc(2,2)
    m2 = gsl_matrix_complex_alloc(2,2)
    m3 = gsl_matrix_complex_alloc(2,2)
    Nalpha = 0.1 + 0.2j
    Nbeta = 0.3 + 0.4j
    Nm1 = array([[1+2j,3+4j],[6+4j,5+6j]])
    Nm2 = array([[3+2j,3+5j],[3+2j,7+4j]])
    Nm3 = array([[1+1j,3+5j],[3+5j,5+8j]])
    alpha = gsl_complex_rect(Nalpha.real, Nalpha.imag)
    beta = gsl_complex_rect(Nbeta.real, Nbeta.imag)
    ary = []
    for i in range(2):
        for j in range(2):
            z = gsl_complex_rect(Nm1[i,j].real, Nm1[i,j].imag)
            gsl_matrix_complex_set(m1, i, j, z)
            z = gsl_complex_rect(Nm2[i,j].real, Nm2[i,j].imag)
            gsl_matrix_complex_set(m2, i, j, z)

    '''
    CBLAS_UPLO_t = CblasUpper, CblasLower
    CBLAS_TRANSPOSE_t = CblasNoTrans, CblasTrans
    There are 4 cases
    '''

    # 1) 11
    for i in range(2):
        for j in range(i,2):
            z = gsl_complex_rect(Nm3[i,j].real, Nm3[i,j].imag)
            gsl_matrix_complex_set(m3, i, j, z)
    Nm4 = Nalpha * dot(Nm1, transpose(Nm2)) + \
            Nalpha * dot(Nm2, transpose(Nm1)) + Nbeta * Nm3
    gsl_blas_zsyr2k(CblasUpper, CblasNoTrans, alpha, m1, m2, beta, m3)
    for i in range(2):
        for j in range(i, 2):
            z = gsl_matrix_complex_get(m3,i,j)
            ary.append(GSL_REAL(z) - Nm4[i,j].real)
            ary.append(GSL_IMAG(z) - Nm4[i,j].imag)

    # 2) 12
    for i in range(2):
        for j in range(i,2):
            z = gsl_complex_rect(Nm3[i,j].real, Nm3[i,j].imag)
            gsl_matrix_complex_set(m3, i, j, z)
    Nm4 = Nalpha * dot(transpose(Nm1), Nm2) + \
            Nalpha * dot(transpose(Nm2), Nm1) + Nbeta * Nm3
    gsl_blas_zsyr2k(CblasUpper, CblasTrans, alpha, m1, m2, beta, m3)
    for i in range(2):
        for j in range(i, 2):
            z = gsl_matrix_complex_get(m3,i,j)
            ary.append(GSL_REAL(z) - Nm4[i,j].real)
            ary.append(GSL_IMAG(z) - Nm4[i,j].imag)

    # 3) 21
    gsl_matrix_complex_set_zero(m3)
    for i in range(2):
        for j in range(0,i+1):
            z = gsl_complex_rect(Nm3[i,j].real, Nm3[i,j].imag)
            gsl_matrix_complex_set(m3, i, j, z)
    Nm4 = Nalpha * dot(Nm1, transpose(Nm2)) + \
            Nalpha * dot(Nm2, transpose(Nm1)) + Nbeta * Nm3
    gsl_blas_zsyr2k(CblasLower, CblasNoTrans, alpha, m1, m2, beta, m3)
    for i in range(2):
        for j in range(0,i+1):
            z = gsl_matrix_complex_get(m3,i,j)
            ary.append(GSL_REAL(z) - Nm4[i,j].real)
            ary.append(GSL_IMAG(z) - Nm4[i,j].imag)

    # 4) 22
    for i in range(2):
        for j in range(0,i+1):
            z = gsl_complex_rect(Nm3[i,j].real, Nm3[i,j].imag)
            gsl_matrix_complex_set(m3, i, j, z)
    Nm4 = Nalpha * dot(transpose(Nm1), Nm2) + \
            Nalpha * dot(transpose(Nm2), Nm1) + Nbeta * Nm3
    gsl_blas_zsyr2k(CblasLower, CblasTrans, alpha, m1, m2, beta, m3)
    for i in range(2):
        for j in range(0,i+1):
            z = gsl_matrix_complex_get(m3,i,j)
            ary.append(GSL_REAL(z) - Nm4[i,j].real)
            ary.append(GSL_IMAG(z) - Nm4[i,j].imag)

    gsl_matrix_complex_free(m1)
    gsl_matrix_complex_free(m2)
    gsl_matrix_complex_free(m3)
    return ary

def t_gsl_blas_ztrmm():
    cdef gsl_matrix_complex * m1, * m2
    cdef gsl_complex z, alpha
    m1 = gsl_matrix_complex_alloc(2,2)
    m2 = gsl_matrix_complex_alloc(2,2)
    Nalpha = 0.1 + 0.2j
    alpha = gsl_complex_rect(Nalpha.real, Nalpha.imag)
    ary = []
    '''
    CBLAS_UPLO_t = CblasUpper, CblasLower
    CBLAS_DIAG_t = CblasUnit, CblasNonUnit
    CBLAS_SIDE_t = CblasLeft, CblasRight
    CBLAS_TRANSPOSE_t = CblasNoTrans, CblasTrans, CblasConjTrans
    There are 24 cases;
    '''

    # upper triangular non unit
    Nm1 = array([[1+2j,3+4j],[0,5+6j]])
    Nm2 = array([[3+2j,3+5j],[3+2j,7+4j]])
    for i in range(2):
        for j in range(2):
            z = gsl_complex_rect(Nm1[i,j].real, Nm1[i,j].imag)
            gsl_matrix_complex_set(m1, i, j, z)
            z = gsl_complex_rect(Nm2[i,j].real, Nm2[i,j].imag)
            gsl_matrix_complex_set(m2, i, j, z)

    # 1) 1211
    Nm3 = Nalpha * dot(Nm1, Nm2)
    gsl_blas_ztrmm(CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, alpha, m1, m2)
    for i in range(2):
        for j in range(i, 2):
            z = gsl_matrix_complex_get(m2,i,j)
            ary.append(GSL_REAL(z) - Nm3[i,j].real)
            ary.append(GSL_IMAG(z) - Nm3[i,j].imag)

    # 2) 1221
    for i in range(2):
        for j in range(2):
            z = gsl_complex_rect(Nm2[i,j].real, Nm2[i,j].imag)
            gsl_matrix_complex_set(m2, i, j, z)
    Nm3 = Nalpha * dot(Nm2, Nm1)
    gsl_blas_ztrmm(CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit, alpha, m1, m2)
    for i in range(2):
        for j in range(i, 2):
            z = gsl_matrix_complex_get(m2,i,j)
            ary.append(GSL_REAL(z) - Nm3[i,j].real)
            ary.append(GSL_IMAG(z) - Nm3[i,j].imag)

    # 3) 1212
    for i in range(2):
        for j in range(2):
            z = gsl_complex_rect(Nm2[i,j].real, Nm2[i,j].imag)
            gsl_matrix_complex_set(m2, i, j, z)
    Nm3 = Nalpha * dot(transpose(Nm1), Nm2)
    gsl_blas_ztrmm(CblasLeft, CblasUpper, CblasTrans, CblasNonUnit, alpha, m1, m2)
    for i in range(2):
        for j in range(i, 2):
            z = gsl_matrix_complex_get(m2,i,j)
            ary.append(GSL_REAL(z) - Nm3[i,j].real)
            ary.append(GSL_IMAG(z) - Nm3[i,j].imag)

    # 4) 1213
    for i in range(2):
        for j in range(2):
            z = gsl_complex_rect(Nm2[i,j].real, Nm2[i,j].imag)
            gsl_matrix_complex_set(m2, i, j, z)
    Nm3 = Nalpha * dot(conjugate(transpose(Nm1)), Nm2)
    gsl_blas_ztrmm(CblasLeft, CblasUpper, CblasConjTrans, CblasNonUnit, alpha, m1, m2)
    for i in range(2):
        for j in range(i, 2):
            z = gsl_matrix_complex_get(m2,i,j)
            ary.append(GSL_REAL(z) - Nm3[i,j].real)
            ary.append(GSL_IMAG(z) - Nm3[i,j].imag)

    # 5) 1223
    for i in range(2):
        for j in range(2):
            z = gsl_complex_rect(Nm2[i,j].real, Nm2[i,j].imag)
            gsl_matrix_complex_set(m2, i, j, z)
    Nm3 = Nalpha * dot(Nm2, conjugate(transpose(Nm1)))
    gsl_blas_ztrmm(CblasRight, CblasUpper, CblasConjTrans, CblasNonUnit, alpha, m1, m2)
    for i in range(2):
        for j in range(i, 2):
            z = gsl_matrix_complex_get(m2,i,j)
            ary.append(GSL_REAL(z) - Nm3[i,j].real)
            ary.append(GSL_IMAG(z) - Nm3[i,j].imag)

    # 6) 1222
    for i in range(2):
        for j in range(2):
            z = gsl_complex_rect(Nm2[i,j].real, Nm2[i,j].imag)
            gsl_matrix_complex_set(m2, i, j, z)
    Nm3 = Nalpha * dot(Nm2, transpose(Nm1))
    gsl_blas_ztrmm(CblasRight, CblasUpper, CblasTrans, CblasNonUnit, alpha, m1, m2)
    for i in range(2):
        for j in range(i, 2):
            z = gsl_matrix_complex_get(m2,i,j)
            ary.append(GSL_REAL(z) - Nm3[i,j].real)
            ary.append(GSL_IMAG(z) - Nm3[i,j].imag)

    # upper triangular unit
    Nm1 = array([[1,3+4j],[0,1]])
    Nm2 = array([[3+2j,3+5j],[3+2j,7+4j]])
    for i in range(2):
        for j in range(2):
            z = gsl_complex_rect(Nm1[i,j].real, Nm1[i,j].imag)
            gsl_matrix_complex_set(m1, i, j, z)
            z = gsl_complex_rect(Nm2[i,j].real, Nm2[i,j].imag)
            gsl_matrix_complex_set(m2, i, j, z)

    # 7) 1111
    Nm3 = Nalpha * dot(Nm1, Nm2)
    gsl_blas_ztrmm(CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, alpha, m1, m2)
    for i in range(2):
        for j in range(i, 2):
            z = gsl_matrix_complex_get(m2,i,j)
            ary.append(GSL_REAL(z) - Nm3[i,j].real)
            ary.append(GSL_IMAG(z) - Nm3[i,j].imag)

    # 8) 1121
    for i in range(2):
        for j in range(2):
            z = gsl_complex_rect(Nm2[i,j].real, Nm2[i,j].imag)
            gsl_matrix_complex_set(m2, i, j, z)
    Nm3 = Nalpha * dot(Nm2, Nm1)
    gsl_blas_ztrmm(CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit, alpha, m1, m2)
    for i in range(2):
        for j in range(i, 2):
            z = gsl_matrix_complex_get(m2,i,j)
            ary.append(GSL_REAL(z) - Nm3[i,j].real)
            ary.append(GSL_IMAG(z) - Nm3[i,j].imag)

    # 9) 1112
    for i in range(2):
        for j in range(2):
            z = gsl_complex_rect(Nm2[i,j].real, Nm2[i,j].imag)
            gsl_matrix_complex_set(m2, i, j, z)
    Nm3 = Nalpha * dot(transpose(Nm1), Nm2)
    gsl_blas_ztrmm(CblasLeft, CblasUpper, CblasTrans, CblasNonUnit, alpha, m1, m2)
    for i in range(2):
        for j in range(i, 2):
            z = gsl_matrix_complex_get(m2,i,j)
            ary.append(GSL_REAL(z) - Nm3[i,j].real)
            ary.append(GSL_IMAG(z) - Nm3[i,j].imag)

    # 10) 1113
    for i in range(2):
        for j in range(2):
            z = gsl_complex_rect(Nm2[i,j].real, Nm2[i,j].imag)
            gsl_matrix_complex_set(m2, i, j, z)
    Nm3 = Nalpha * dot(conjugate(transpose(Nm1)), Nm2)
    gsl_blas_ztrmm(CblasLeft, CblasUpper, CblasConjTrans, CblasNonUnit, alpha, m1, m2)
    for i in range(2):
        for j in range(i, 2):
            z = gsl_matrix_complex_get(m2,i,j)
            ary.append(GSL_REAL(z) - Nm3[i,j].real)
            ary.append(GSL_IMAG(z) - Nm3[i,j].imag)

    # 11) 1123
    for i in range(2):
        for j in range(2):
            z = gsl_complex_rect(Nm2[i,j].real, Nm2[i,j].imag)
            gsl_matrix_complex_set(m2, i, j, z)
    Nm3 = Nalpha * dot(Nm2, conjugate(transpose(Nm1)))
    gsl_blas_ztrmm(CblasRight, CblasUpper, CblasConjTrans, CblasNonUnit, alpha, m1, m2)
    for i in range(2):
        for j in range(i, 2):
            z = gsl_matrix_complex_get(m2,i,j)
            ary.append(GSL_REAL(z) - Nm3[i,j].real)
            ary.append(GSL_IMAG(z) - Nm3[i,j].imag)

    # 12) 1122
    for i in range(2):
        for j in range(2):
            z = gsl_complex_rect(Nm2[i,j].real, Nm2[i,j].imag)
            gsl_matrix_complex_set(m2, i, j, z)
    Nm3 = Nalpha * dot(Nm2, transpose(Nm1))
    gsl_blas_ztrmm(CblasRight, CblasUpper, CblasTrans, CblasNonUnit, alpha, m1, m2)
    for i in range(2):
        for j in range(i, 2):
            z = gsl_matrix_complex_get(m2,i,j)
            ary.append(GSL_REAL(z) - Nm3[i,j].real)
            ary.append(GSL_IMAG(z) - Nm3[i,j].imag)

    # lower triangular non unit
    Nm1 = array([[1+2j,0],[3 + 4j,5+6j]])
    Nm2 = array([[3+2j,3+5j],[3+2j,7+4j]])
    for i in range(2):
        for j in range(2):
            z = gsl_complex_rect(Nm1[i,j].real, Nm1[i,j].imag)
            gsl_matrix_complex_set(m1, i, j, z)
            z = gsl_complex_rect(Nm2[i,j].real, Nm2[i,j].imag)
            gsl_matrix_complex_set(m2, i, j, z)

    # 13) 2211
    Nm3 = Nalpha * dot(Nm1, Nm2)
    gsl_blas_ztrmm(CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit, alpha, m1, m2)
    for i in range(2):
        for j in range(i, 2):
            z = gsl_matrix_complex_get(m2,i,j)
            ary.append(GSL_REAL(z) - Nm3[i,j].real)
            ary.append(GSL_IMAG(z) - Nm3[i,j].imag)

    # 14) 2221
    for i in range(2):
        for j in range(2):
            z = gsl_complex_rect(Nm2[i,j].real, Nm2[i,j].imag)
            gsl_matrix_complex_set(m2, i, j, z)
    Nm3 = Nalpha * dot(Nm2, Nm1)
    gsl_blas_ztrmm(CblasRight, CblasLower, CblasNoTrans, CblasNonUnit, alpha, m1, m2)
    for i in range(2):
        for j in range(i, 2):
            z = gsl_matrix_complex_get(m2,i,j)
            ary.append(GSL_REAL(z) - Nm3[i,j].real)
            ary.append(GSL_IMAG(z) - Nm3[i,j].imag)

    # 15) 2212
    for i in range(2):
        for j in range(2):
            z = gsl_complex_rect(Nm2[i,j].real, Nm2[i,j].imag)
            gsl_matrix_complex_set(m2, i, j, z)
    Nm3 = Nalpha * dot(transpose(Nm1), Nm2)
    gsl_blas_ztrmm(CblasLeft, CblasLower, CblasTrans, CblasNonUnit, alpha, m1, m2)
    for i in range(2):
        for j in range(i, 2):
            z = gsl_matrix_complex_get(m2,i,j)
            ary.append(GSL_REAL(z) - Nm3[i,j].real)
            ary.append(GSL_IMAG(z) - Nm3[i,j].imag)

    # 16) 2213
    for i in range(2):
        for j in range(2):
            z = gsl_complex_rect(Nm2[i,j].real, Nm2[i,j].imag)
            gsl_matrix_complex_set(m2, i, j, z)
    Nm3 = Nalpha * dot(conjugate(transpose(Nm1)), Nm2)
    gsl_blas_ztrmm(CblasLeft, CblasLower, CblasConjTrans, CblasNonUnit, alpha, m1, m2)
    for i in range(2):
        for j in range(i, 2):
            z = gsl_matrix_complex_get(m2,i,j)
            ary.append(GSL_REAL(z) - Nm3[i,j].real)
            ary.append(GSL_IMAG(z) - Nm3[i,j].imag)

    # 17) 2223
    for i in range(2):
        for j in range(2):
            z = gsl_complex_rect(Nm2[i,j].real, Nm2[i,j].imag)
            gsl_matrix_complex_set(m2, i, j, z)
    Nm3 = Nalpha * dot(Nm2, conjugate(transpose(Nm1)))
    gsl_blas_ztrmm(CblasRight, CblasLower, CblasConjTrans, CblasNonUnit, alpha, m1, m2)
    for i in range(2):
        for j in range(i, 2):
            z = gsl_matrix_complex_get(m2,i,j)
            ary.append(GSL_REAL(z) - Nm3[i,j].real)
            ary.append(GSL_IMAG(z) - Nm3[i,j].imag)

    # 18) 2222
    for i in range(2):
        for j in range(2):
            z = gsl_complex_rect(Nm2[i,j].real, Nm2[i,j].imag)
            gsl_matrix_complex_set(m2, i, j, z)
    Nm3 = Nalpha * dot(Nm2, transpose(Nm1))
    gsl_blas_ztrmm(CblasRight, CblasLower, CblasTrans, CblasNonUnit, alpha, m1, m2)
    for i in range(2):
        for j in range(i, 2):
            z = gsl_matrix_complex_get(m2,i,j)
            ary.append(GSL_REAL(z) - Nm3[i,j].real)
            ary.append(GSL_IMAG(z) - Nm3[i,j].imag)

    # lower triangular unit
    Nm1 = array([[1,0],[3+4j,1]])
    Nm2 = array([[3+2j,3+5j],[3+2j,7+4j]])
    for i in range(2):
        for j in range(2):
            z = gsl_complex_rect(Nm1[i,j].real, Nm1[i,j].imag)
            gsl_matrix_complex_set(m1, i, j, z)
            z = gsl_complex_rect(Nm2[i,j].real, Nm2[i,j].imag)
            gsl_matrix_complex_set(m2, i, j, z)

    # 19) 2111
    Nm3 = Nalpha * dot(Nm1, Nm2)
    gsl_blas_ztrmm(CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit, alpha, m1, m2)
    for i in range(2):
        for j in range(i, 2):
            z = gsl_matrix_complex_get(m2,i,j)
            ary.append(GSL_REAL(z) - Nm3[i,j].real)
            ary.append(GSL_IMAG(z) - Nm3[i,j].imag)

    # 20) 2121
    for i in range(2):
        for j in range(2):
            z = gsl_complex_rect(Nm2[i,j].real, Nm2[i,j].imag)
            gsl_matrix_complex_set(m2, i, j, z)
    Nm3 = Nalpha * dot(Nm2, Nm1)
    gsl_blas_ztrmm(CblasRight, CblasLower, CblasNoTrans, CblasNonUnit, alpha, m1, m2)
    for i in range(2):
        for j in range(i, 2):
            z = gsl_matrix_complex_get(m2,i,j)
            ary.append(GSL_REAL(z) - Nm3[i,j].real)
            ary.append(GSL_IMAG(z) - Nm3[i,j].imag)

    # 21) 2112
    for i in range(2):
        for j in range(2):
            z = gsl_complex_rect(Nm2[i,j].real, Nm2[i,j].imag)
            gsl_matrix_complex_set(m2, i, j, z)
    Nm3 = Nalpha * dot(transpose(Nm1), Nm2)
    gsl_blas_ztrmm(CblasLeft, CblasLower, CblasTrans, CblasNonUnit, alpha, m1, m2)
    for i in range(2):
        for j in range(i, 2):
            z = gsl_matrix_complex_get(m2,i,j)
            ary.append(GSL_REAL(z) - Nm3[i,j].real)
            ary.append(GSL_IMAG(z) - Nm3[i,j].imag)

    # 22) 2113
    for i in range(2):
        for j in range(2):
            z = gsl_complex_rect(Nm2[i,j].real, Nm2[i,j].imag)
            gsl_matrix_complex_set(m2, i, j, z)
    Nm3 = Nalpha * dot(conjugate(transpose(Nm1)), Nm2)
    gsl_blas_ztrmm(CblasLeft, CblasLower, CblasConjTrans, CblasNonUnit, alpha, m1, m2)
    for i in range(2):
        for j in range(i, 2):
            z = gsl_matrix_complex_get(m2,i,j)
            ary.append(GSL_REAL(z) - Nm3[i,j].real)
            ary.append(GSL_IMAG(z) - Nm3[i,j].imag)

    # 23) 2123
    for i in range(2):
        for j in range(2):
            z = gsl_complex_rect(Nm2[i,j].real, Nm2[i,j].imag)
            gsl_matrix_complex_set(m2, i, j, z)
    Nm3 = Nalpha * dot(Nm2, conjugate(transpose(Nm1)))
    gsl_blas_ztrmm(CblasRight, CblasLower, CblasConjTrans, CblasNonUnit, alpha, m1, m2)
    for i in range(2):
        for j in range(i, 2):
            z = gsl_matrix_complex_get(m2,i,j)
            ary.append(GSL_REAL(z) - Nm3[i,j].real)
            ary.append(GSL_IMAG(z) - Nm3[i,j].imag)

    # 24) 2122
    for i in range(2):
        for j in range(2):
            z = gsl_complex_rect(Nm2[i,j].real, Nm2[i,j].imag)
            gsl_matrix_complex_set(m2, i, j, z)
    Nm3 = Nalpha * dot(Nm2, transpose(Nm1))
    gsl_blas_ztrmm(CblasRight, CblasLower, CblasTrans, CblasNonUnit, alpha, m1, m2)
    for i in range(2):
        for j in range(i, 2):
            z = gsl_matrix_complex_get(m2,i,j)
            ary.append(GSL_REAL(z) - Nm3[i,j].real)
            ary.append(GSL_IMAG(z) - Nm3[i,j].imag)

    gsl_matrix_complex_free(m1)
    gsl_matrix_complex_free(m2)
    return ary

def t_gsl_blas_ztrsm():
    cdef gsl_matrix_complex * m1, * m2
    cdef gsl_complex z, alpha
    m1 = gsl_matrix_complex_alloc(2,2)
    m2 = gsl_matrix_complex_alloc(2,2)
    Nalpha = 0.1 + 0.2j
    alpha = gsl_complex_rect(Nalpha.real, Nalpha.imag)
    ary = []
    '''
    CBLAS_UPLO_t = CblasUpper, CblasLower
    CBLAS_DIAG_t = CblasUnit, CblasNonUnit
    CBLAS_SIDE_t = CblasLeft, CblasRight
    CBLAS_TRANSPOSE_t = CblasNoTrans, CblasTrans, CblasConjTrans
    There are 24 cases;
    '''

    # upper triangular non unit
    Nm1 = array([[1+2j,3+4j],[0,5+6j]])
    Nm2 = array([[3+2j,3+5j],[3+2j,7+4j]])
    for i in range(2):
        for j in range(2):
            z = gsl_complex_rect(Nm1[i,j].real, Nm1[i,j].imag)
            gsl_matrix_complex_set(m1, i, j, z)
            z = gsl_complex_rect(Nm2[i,j].real, Nm2[i,j].imag)
            gsl_matrix_complex_set(m2, i, j, z)

    # 1) 1211
    Nm3 = Nalpha * dot(inv(Nm1), Nm2)
    gsl_blas_ztrsm(CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, alpha, m1, m2)
    for i in range(2):
        for j in range(i, 2):
            z = gsl_matrix_complex_get(m2,i,j)
            ary.append(GSL_REAL(z) - Nm3[i,j].real)
            ary.append(GSL_IMAG(z) - Nm3[i,j].imag)

    # 2) 1221
    for i in range(2):
        for j in range(2):
            z = gsl_complex_rect(Nm2[i,j].real, Nm2[i,j].imag)
            gsl_matrix_complex_set(m2, i, j, z)
    Nm3 = Nalpha * dot(Nm2, inv(Nm1))
    gsl_blas_ztrsm(CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit, alpha, m1, m2)
    for i in range(2):
        for j in range(i, 2):
            z = gsl_matrix_complex_get(m2,i,j)
            ary.append(GSL_REAL(z) - Nm3[i,j].real)
            ary.append(GSL_IMAG(z) - Nm3[i,j].imag)

    # 3) 1212
    for i in range(2):
        for j in range(2):
            z = gsl_complex_rect(Nm2[i,j].real, Nm2[i,j].imag)
            gsl_matrix_complex_set(m2, i, j, z)
    Nm3 = Nalpha * dot(transpose(inv(Nm1)), Nm2)
    gsl_blas_ztrsm(CblasLeft, CblasUpper, CblasTrans, CblasNonUnit, alpha, m1, m2)
    for i in range(2):
        for j in range(i, 2):
            z = gsl_matrix_complex_get(m2,i,j)
            ary.append(GSL_REAL(z) - Nm3[i,j].real)
            ary.append(GSL_IMAG(z) - Nm3[i,j].imag)

    # 4) 1213
    for i in range(2):
        for j in range(2):
            z = gsl_complex_rect(Nm2[i,j].real, Nm2[i,j].imag)
            gsl_matrix_complex_set(m2, i, j, z)
    Nm3 = Nalpha * dot(conjugate(transpose(inv(Nm1))), Nm2)
    gsl_blas_ztrsm(CblasLeft, CblasUpper, CblasConjTrans, CblasNonUnit, alpha, m1, m2)
    for i in range(2):
        for j in range(i, 2):
            z = gsl_matrix_complex_get(m2,i,j)
            ary.append(GSL_REAL(z) - Nm3[i,j].real)
            ary.append(GSL_IMAG(z) - Nm3[i,j].imag)

    # 5) 1223
    for i in range(2):
        for j in range(2):
            z = gsl_complex_rect(Nm2[i,j].real, Nm2[i,j].imag)
            gsl_matrix_complex_set(m2, i, j, z)
    Nm3 = Nalpha * dot(Nm2, conjugate(transpose(inv(Nm1))))
    gsl_blas_ztrsm(CblasRight, CblasUpper, CblasConjTrans, CblasNonUnit, alpha, m1, m2)
    for i in range(2):
        for j in range(i, 2):
            z = gsl_matrix_complex_get(m2,i,j)
            ary.append(GSL_REAL(z) - Nm3[i,j].real)
            ary.append(GSL_IMAG(z) - Nm3[i,j].imag)

    # 6) 1222
    for i in range(2):
        for j in range(2):
            z = gsl_complex_rect(Nm2[i,j].real, Nm2[i,j].imag)
            gsl_matrix_complex_set(m2, i, j, z)
    Nm3 = Nalpha * dot(Nm2, transpose(inv(Nm1)))
    gsl_blas_ztrsm(CblasRight, CblasUpper, CblasTrans, CblasNonUnit, alpha, m1, m2)
    for i in range(2):
        for j in range(i, 2):
            z = gsl_matrix_complex_get(m2,i,j)
            ary.append(GSL_REAL(z) - Nm3[i,j].real)
            ary.append(GSL_IMAG(z) - Nm3[i,j].imag)

    # upper triangular unit
    Nm1 = array([[1,3+4j],[0,1]])
    Nm2 = array([[3+2j,3+5j],[3+2j,7+4j]])
    for i in range(2):
        for j in range(2):
            z = gsl_complex_rect(Nm1[i,j].real, Nm1[i,j].imag)
            gsl_matrix_complex_set(m1, i, j, z)
            z = gsl_complex_rect(Nm2[i,j].real, Nm2[i,j].imag)
            gsl_matrix_complex_set(m2, i, j, z)

    # 7) 1111
    Nm3 = Nalpha * dot(inv(Nm1), Nm2)
    gsl_blas_ztrsm(CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, alpha, m1, m2)
    for i in range(2):
        for j in range(i, 2):
            z = gsl_matrix_complex_get(m2,i,j)
            ary.append(GSL_REAL(z) - Nm3[i,j].real)
            ary.append(GSL_IMAG(z) - Nm3[i,j].imag)

    # 8) 1121
    for i in range(2):
        for j in range(2):
            z = gsl_complex_rect(Nm2[i,j].real, Nm2[i,j].imag)
            gsl_matrix_complex_set(m2, i, j, z)
    Nm3 = Nalpha * dot(Nm2, inv(Nm1))
    gsl_blas_ztrsm(CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit, alpha, m1, m2)
    for i in range(2):
        for j in range(i, 2):
            z = gsl_matrix_complex_get(m2,i,j)
            ary.append(GSL_REAL(z) - Nm3[i,j].real)
            ary.append(GSL_IMAG(z) - Nm3[i,j].imag)

    # 9) 1112
    for i in range(2):
        for j in range(2):
            z = gsl_complex_rect(Nm2[i,j].real, Nm2[i,j].imag)
            gsl_matrix_complex_set(m2, i, j, z)
    Nm3 = Nalpha * dot(transpose(inv(Nm1)), Nm2)
    gsl_blas_ztrsm(CblasLeft, CblasUpper, CblasTrans, CblasNonUnit, alpha, m1, m2)
    for i in range(2):
        for j in range(i, 2):
            z = gsl_matrix_complex_get(m2,i,j)
            ary.append(GSL_REAL(z) - Nm3[i,j].real)
            ary.append(GSL_IMAG(z) - Nm3[i,j].imag)

    # 10) 1113
    for i in range(2):
        for j in range(2):
            z = gsl_complex_rect(Nm2[i,j].real, Nm2[i,j].imag)
            gsl_matrix_complex_set(m2, i, j, z)
    Nm3 = Nalpha * dot(conjugate(transpose(inv(Nm1))), Nm2)
    gsl_blas_ztrsm(CblasLeft, CblasUpper, CblasConjTrans, CblasNonUnit, alpha, m1, m2)
    for i in range(2):
        for j in range(i, 2):
            z = gsl_matrix_complex_get(m2,i,j)
            ary.append(GSL_REAL(z) - Nm3[i,j].real)
            ary.append(GSL_IMAG(z) - Nm3[i,j].imag)

    # 11) 1123
    for i in range(2):
        for j in range(2):
            z = gsl_complex_rect(Nm2[i,j].real, Nm2[i,j].imag)
            gsl_matrix_complex_set(m2, i, j, z)
    Nm3 = Nalpha * dot(Nm2, conjugate(transpose(inv(Nm1))))
    gsl_blas_ztrsm(CblasRight, CblasUpper, CblasConjTrans, CblasNonUnit, alpha, m1, m2)
    for i in range(2):
        for j in range(i, 2):
            z = gsl_matrix_complex_get(m2,i,j)
            ary.append(GSL_REAL(z) - Nm3[i,j].real)
            ary.append(GSL_IMAG(z) - Nm3[i,j].imag)

    # 12) 1122
    for i in range(2):
        for j in range(2):
            z = gsl_complex_rect(Nm2[i,j].real, Nm2[i,j].imag)
            gsl_matrix_complex_set(m2, i, j, z)
    Nm3 = Nalpha * dot(Nm2, transpose(inv(Nm1)))
    gsl_blas_ztrsm(CblasRight, CblasUpper, CblasTrans, CblasNonUnit, alpha, m1, m2)
    for i in range(2):
        for j in range(i, 2):
            z = gsl_matrix_complex_get(m2,i,j)
            ary.append(GSL_REAL(z) - Nm3[i,j].real)
            ary.append(GSL_IMAG(z) - Nm3[i,j].imag)

    # lower triangular non unit
    Nm1 = array([[1+2j,0],[3 + 4j,5+6j]])
    Nm2 = array([[3+2j,3+5j],[3+2j,7+4j]])
    for i in range(2):
        for j in range(2):
            z = gsl_complex_rect(Nm1[i,j].real, Nm1[i,j].imag)
            gsl_matrix_complex_set(m1, i, j, z)
            z = gsl_complex_rect(Nm2[i,j].real, Nm2[i,j].imag)
            gsl_matrix_complex_set(m2, i, j, z)

    # 13) 2211
    Nm3 = Nalpha * dot(inv(Nm1), Nm2)
    gsl_blas_ztrsm(CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit, alpha, m1, m2)
    for i in range(2):
        for j in range(i, 2):
            z = gsl_matrix_complex_get(m2,i,j)
            ary.append(GSL_REAL(z) - Nm3[i,j].real)
            ary.append(GSL_IMAG(z) - Nm3[i,j].imag)

    # 14) 2221
    for i in range(2):
        for j in range(2):
            z = gsl_complex_rect(Nm2[i,j].real, Nm2[i,j].imag)
            gsl_matrix_complex_set(m2, i, j, z)
    Nm3 = Nalpha * dot(Nm2, inv(Nm1))
    gsl_blas_ztrsm(CblasRight, CblasLower, CblasNoTrans, CblasNonUnit, alpha, m1, m2)
    for i in range(2):
        for j in range(i, 2):
            z = gsl_matrix_complex_get(m2,i,j)
            ary.append(GSL_REAL(z) - Nm3[i,j].real)
            ary.append(GSL_IMAG(z) - Nm3[i,j].imag)

    # 15) 2212
    for i in range(2):
        for j in range(2):
            z = gsl_complex_rect(Nm2[i,j].real, Nm2[i,j].imag)
            gsl_matrix_complex_set(m2, i, j, z)
    Nm3 = Nalpha * dot(transpose(inv(Nm1)), Nm2)
    gsl_blas_ztrsm(CblasLeft, CblasLower, CblasTrans, CblasNonUnit, alpha, m1, m2)
    for i in range(2):
        for j in range(i, 2):
            z = gsl_matrix_complex_get(m2,i,j)
            ary.append(GSL_REAL(z) - Nm3[i,j].real)
            ary.append(GSL_IMAG(z) - Nm3[i,j].imag)

    # 16) 2213
    for i in range(2):
        for j in range(2):
            z = gsl_complex_rect(Nm2[i,j].real, Nm2[i,j].imag)
            gsl_matrix_complex_set(m2, i, j, z)
    Nm3 = Nalpha * dot(conjugate(transpose(inv(Nm1))), Nm2)
    gsl_blas_ztrsm(CblasLeft, CblasLower, CblasConjTrans, CblasNonUnit, alpha, m1, m2)
    for i in range(2):
        for j in range(i, 2):
            z = gsl_matrix_complex_get(m2,i,j)
            ary.append(GSL_REAL(z) - Nm3[i,j].real)
            ary.append(GSL_IMAG(z) - Nm3[i,j].imag)

    # 17) 2223
    for i in range(2):
        for j in range(2):
            z = gsl_complex_rect(Nm2[i,j].real, Nm2[i,j].imag)
            gsl_matrix_complex_set(m2, i, j, z)
    Nm3 = Nalpha * dot(Nm2, conjugate(transpose(inv(Nm1))))
    gsl_blas_ztrsm(CblasRight, CblasLower, CblasConjTrans, CblasNonUnit, alpha, m1, m2)
    for i in range(2):
        for j in range(i, 2):
            z = gsl_matrix_complex_get(m2,i,j)
            ary.append(GSL_REAL(z) - Nm3[i,j].real)
            ary.append(GSL_IMAG(z) - Nm3[i,j].imag)

    # 18) 2222
    for i in range(2):
        for j in range(2):
            z = gsl_complex_rect(Nm2[i,j].real, Nm2[i,j].imag)
            gsl_matrix_complex_set(m2, i, j, z)
    Nm3 = Nalpha * dot(Nm2, transpose(inv(Nm1)))
    gsl_blas_ztrsm(CblasRight, CblasLower, CblasTrans, CblasNonUnit, alpha, m1, m2)
    for i in range(2):
        for j in range(i, 2):
            z = gsl_matrix_complex_get(m2,i,j)
            ary.append(GSL_REAL(z) - Nm3[i,j].real)
            ary.append(GSL_IMAG(z) - Nm3[i,j].imag)

    # lower triangular unit
    Nm1 = array([[1,0],[3+4j,1]])
    Nm2 = array([[3+2j,3+5j],[3+2j,7+4j]])
    for i in range(2):
        for j in range(2):
            z = gsl_complex_rect(Nm1[i,j].real, Nm1[i,j].imag)
            gsl_matrix_complex_set(m1, i, j, z)
            z = gsl_complex_rect(Nm2[i,j].real, Nm2[i,j].imag)
            gsl_matrix_complex_set(m2, i, j, z)

    # 19) 2111
    Nm3 = Nalpha * dot(inv(Nm1), Nm2)
    gsl_blas_ztrsm(CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit, alpha, m1, m2)
    for i in range(2):
        for j in range(i, 2):
            z = gsl_matrix_complex_get(m2,i,j)
            ary.append(GSL_REAL(z) - Nm3[i,j].real)
            ary.append(GSL_IMAG(z) - Nm3[i,j].imag)

    # 20) 2121
    for i in range(2):
        for j in range(2):
            z = gsl_complex_rect(Nm2[i,j].real, Nm2[i,j].imag)
            gsl_matrix_complex_set(m2, i, j, z)
    Nm3 = Nalpha * dot(Nm2, inv(Nm1))
    gsl_blas_ztrsm(CblasRight, CblasLower, CblasNoTrans, CblasNonUnit, alpha, m1, m2)
    for i in range(2):
        for j in range(i, 2):
            z = gsl_matrix_complex_get(m2,i,j)
            ary.append(GSL_REAL(z) - Nm3[i,j].real)
            ary.append(GSL_IMAG(z) - Nm3[i,j].imag)

    # 21) 2112
    for i in range(2):
        for j in range(2):
            z = gsl_complex_rect(Nm2[i,j].real, Nm2[i,j].imag)
            gsl_matrix_complex_set(m2, i, j, z)
    Nm3 = Nalpha * dot(transpose(inv(Nm1)), Nm2)
    gsl_blas_ztrsm(CblasLeft, CblasLower, CblasTrans, CblasNonUnit, alpha, m1, m2)
    for i in range(2):
        for j in range(i, 2):
            z = gsl_matrix_complex_get(m2,i,j)
            ary.append(GSL_REAL(z) - Nm3[i,j].real)
            ary.append(GSL_IMAG(z) - Nm3[i,j].imag)

    # 22) 2113
    for i in range(2):
        for j in range(2):
            z = gsl_complex_rect(Nm2[i,j].real, Nm2[i,j].imag)
            gsl_matrix_complex_set(m2, i, j, z)
    Nm3 = Nalpha * dot(conjugate(transpose(inv(Nm1))), Nm2)
    gsl_blas_ztrsm(CblasLeft, CblasLower, CblasConjTrans, CblasNonUnit, alpha, m1, m2)
    for i in range(2):
        for j in range(i, 2):
            z = gsl_matrix_complex_get(m2,i,j)
            ary.append(GSL_REAL(z) - Nm3[i,j].real)
            ary.append(GSL_IMAG(z) - Nm3[i,j].imag)

    # 23) 2123
    for i in range(2):
        for j in range(2):
            z = gsl_complex_rect(Nm2[i,j].real, Nm2[i,j].imag)
            gsl_matrix_complex_set(m2, i, j, z)
    Nm3 = Nalpha * dot(Nm2, conjugate(transpose(inv(Nm1))))
    gsl_blas_ztrsm(CblasRight, CblasLower, CblasConjTrans, CblasNonUnit, alpha, m1, m2)
    for i in range(2):
        for j in range(i, 2):
            z = gsl_matrix_complex_get(m2,i,j)
            ary.append(GSL_REAL(z) - Nm3[i,j].real)
            ary.append(GSL_IMAG(z) - Nm3[i,j].imag)

    # 24) 2122
    for i in range(2):
        for j in range(2):
            z = gsl_complex_rect(Nm2[i,j].real, Nm2[i,j].imag)
            gsl_matrix_complex_set(m2, i, j, z)
    Nm3 = Nalpha * dot(Nm2, transpose(inv(Nm1)))
    gsl_blas_ztrsm(CblasRight, CblasLower, CblasTrans, CblasNonUnit, alpha, m1, m2)
    for i in range(2):
        for j in range(i, 2):
            z = gsl_matrix_complex_get(m2,i,j)
            ary.append(GSL_REAL(z) - Nm3[i,j].real)
            ary.append(GSL_IMAG(z) - Nm3[i,j].imag)

    gsl_matrix_complex_free(m1)
    gsl_matrix_complex_free(m2)
    return ary

def t_gsl_blas_zhemm():
    cdef gsl_matrix_complex * m1, * m2, * m3
    cdef gsl_complex z, alpha, beta
    m1 = gsl_matrix_complex_alloc(2,2)
    m2 = gsl_matrix_complex_alloc(2,2)
    m3 = gsl_matrix_complex_alloc(2,2)
    Nalpha = 0.1 + 0.2j
    Nbeta = 0.3 + 0.4j
    Nm1 = array([[1,3+4j],[3-4j,5]])
    Nm2 = array([[3+2j,3+5j],[3+5j,7+6j]])
    Nm3 = array([[1+1j,3+5j],[7+5j,5+8j]])
    alpha = gsl_complex_rect(Nalpha.real, Nalpha.imag)
    beta = gsl_complex_rect(Nbeta.real, Nbeta.imag)
    ary = []
    for i in range(2):
        for j in range(2):
            z = gsl_complex_rect(Nm2[i,j].real, Nm2[i,j].imag)
            gsl_matrix_complex_set(m2, i, j, z)
            z = gsl_complex_rect(Nm3[i,j].real, Nm3[i,j].imag)
            gsl_matrix_complex_set(m3, i, j, z)

    '''
    CBLAS_UPLO_t = CblasUpper, CblasLower
    CBLAS_SIDE_t = CblasLeft, CblasRight
    There are 4 cases
    '''

    # 1) 11
    for i in range(2):
        for j in range(i,2):
            z = gsl_complex_rect(Nm1[i,j].real, Nm1[i,j].imag)
            gsl_matrix_complex_set(m1, i, j, z)
    Nm4 = Nalpha * dot(Nm1,Nm2) + Nbeta * Nm3
    gsl_blas_zhemm(CblasLeft, CblasUpper,alpha, m1, m2, beta, m3)
    for i in range(2):
        for j in range(2):
            z = gsl_matrix_complex_get(m3,i,j)
            ary.append(GSL_REAL(z) - Nm4[i,j].real)
            ary.append(GSL_IMAG(z) - Nm4[i,j].imag)

    # 2) 12
    for i in range(2):
        for j in range(0,2):
            z = gsl_complex_rect(Nm3[i,j].real, Nm3[i,j].imag)
            gsl_matrix_complex_set(m3, i, j, z)
    Nm4 = Nalpha * dot(Nm2,Nm1) + Nbeta * Nm3
    gsl_blas_zhemm(CblasRight, CblasUpper,alpha, m1, m2, beta, m3)
    for i in range(2):
        for j in range(2):
            z = gsl_matrix_complex_get(m3,i,j)
            ary.append(GSL_REAL(z) - Nm4[i,j].real)
            ary.append(GSL_IMAG(z) - Nm4[i,j].imag)

    # 3) 21
    gsl_matrix_complex_set_zero(m1)
    for i in range(2):
        for j in range(0,i+1):
            z = gsl_complex_rect(Nm1[i,j].real, Nm1[i,j].imag)
            gsl_matrix_complex_set(m1, i, j, z)
    for i in range(2):
        for j in range(0,2):
            z = gsl_complex_rect(Nm3[i,j].real, Nm3[i,j].imag)
            gsl_matrix_complex_set(m3, i, j, z)
    Nm4 = Nalpha * dot(Nm1,Nm2) + Nbeta * Nm3
    gsl_blas_zhemm(CblasLeft, CblasLower,alpha, m1, m2, beta, m3)
    for i in range(2):
        for j in range(2):
            z = gsl_matrix_complex_get(m3,i,j)
            ary.append(GSL_REAL(z) - Nm4[i,j].real)
            ary.append(GSL_IMAG(z) - Nm4[i,j].imag)

    # 4) 22
    for i in range(2):
        for j in range(0,2):
            z = gsl_complex_rect(Nm3[i,j].real, Nm3[i,j].imag)
            gsl_matrix_complex_set(m3, i, j, z)
    Nm4 = Nalpha * dot(Nm2,Nm1) + Nbeta * Nm3
    gsl_blas_zhemm(CblasRight, CblasLower,alpha, m1, m2, beta, m3)
    for i in range(2):
        for j in range(2):
            z = gsl_matrix_complex_get(m3,i,j)
            ary.append(GSL_REAL(z) - Nm4[i,j].real)
            ary.append(GSL_IMAG(z) - Nm4[i,j].imag)


    gsl_matrix_complex_free(m1)
    gsl_matrix_complex_free(m2)
    gsl_matrix_complex_free(m3)
    return ary

def t_gsl_blas_zherk():
    cdef gsl_matrix_complex * m1, * m2
    cdef gsl_complex z
    cdef double alpha, beta
    m1 = gsl_matrix_complex_alloc(2,2)
    m2 = gsl_matrix_complex_alloc(2,2)
    alpha = 0.1
    beta = 0.3
    Nm1 = array([[1+2j,3+4j],[5+4j,5+6j]])
    Nm2 = array([[3,3+5j],[3-5j,2]])
    ary = []
    for i in range(2):
        for j in range(2):
            z = gsl_complex_rect(Nm1[i,j].real, Nm1[i,j].imag)
            gsl_matrix_complex_set(m1, i, j, z)

    '''
    CBLAS_UPLO_t = CblasUpper, CblasLower
    CBLAS_TRANSPOSE_t = CblasNoTrans, CblasTrans
    There are 4 cases
    '''

    # upper triangular part of m2 used
    # 1) 11
    for i in range(2):
        for j in range(i,2):
            z = gsl_complex_rect(Nm2[i,j].real, Nm2[i,j].imag)
            gsl_matrix_complex_set(m2, i, j, z)
    Nm3 = alpha * dot(Nm1,conjugate(transpose(Nm1))) + beta * Nm2
    gsl_blas_zherk(CblasUpper, CblasNoTrans,alpha, m1, beta, m2)
    for i in range(2):
        for j in range(i,2):
            z = gsl_matrix_complex_get(m2,i,j)
            ary.append(GSL_REAL(z) - Nm3[i,j].real)
            ary.append(GSL_IMAG(z) - Nm3[i,j].imag)

    # 2) 12
    for i in range(2):
        for j in range(i,2):
            z = gsl_complex_rect(Nm2[i,j].real, Nm2[i,j].imag)
            gsl_matrix_complex_set(m2, i, j, z)
    Nm3 = alpha * dot(conjugate(transpose(Nm1)), Nm1) + beta * Nm2

    gsl_blas_zherk(CblasUpper, CblasNoTrans,alpha, m1, beta, m2)
    # BUG: if I put
    # gsl_blas_zherk(CblasUpper, CblasTrans, alpha, m1, beta, m2)
    # it compiles, but python test1.py gives
    # unrecognized operation

    for i in range(2):
        for j in range(i,2):
            z = gsl_matrix_complex_get(m2,i,j)
            #ary.append(GSL_REAL(z) - Nm3[i,j].real)
            #ary.append(GSL_IMAG(z) - Nm3[i,j].imag)

    # lower triangular part of m2 used
    # 3) 21
    gsl_matrix_complex_set_zero(m2)
    for i in range(2):
        for j in range(0,i+1):
            z = gsl_complex_rect(Nm2[i,j].real, Nm2[i,j].imag)
            gsl_matrix_complex_set(m2, i, j, z)
    Nm3 = alpha * dot(Nm1,conjugate(transpose(Nm1))) + beta * Nm2
    gsl_blas_zherk(CblasLower, CblasNoTrans,alpha, m1, beta, m2)
    for i in range(2):
        for j in range(0,i+1):
            z = gsl_matrix_complex_get(m2,i,j)
            ary.append(GSL_REAL(z) - Nm3[i,j].real)
            ary.append(GSL_IMAG(z) - Nm3[i,j].imag)

    # 4) 22
    for i in range(2):
        for j in range(0,i+1):
            z = gsl_complex_rect(Nm2[i,j].real, Nm2[i,j].imag)
            gsl_matrix_complex_set(m2, i, j, z)
    Nm3 = alpha * dot(conjugate(transpose(Nm1)),Nm1) + beta * Nm2
    gsl_blas_zherk(CblasLower, CblasNoTrans, alpha, m1, beta, m2)
    # BUG: if I put
    # gsl_blas_zherk(CblasLower, CblasTrans, alpha, m1, beta, m2)
    # it compiles, but python test1.py gives
    # unrecognized operation
    for i in range(2):
        for j in range(0,i+1):
            z = gsl_matrix_complex_get(m2,i,j)
            ary.append(GSL_REAL(z) - Nm3[i,j].real)
            ary.append(GSL_IMAG(z) - Nm3[i,j].imag)
    '''
    '''
    gsl_matrix_complex_free(m1)
    gsl_matrix_complex_free(m2)
    return ary
