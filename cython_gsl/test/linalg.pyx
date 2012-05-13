from cython_gsl cimport *
from numpy import array, dot, transpose, outer, identity

cdef gsl_vector_set_from_Num(gsl_vector *v1, Nv1):
    cdef int i
    for i from 0 <= i < Nv1.shape[0]:
        gsl_vector_set(v1, i, Nv1[i])

cdef gsl_matrix_set_from_Num(gsl_matrix * m1, Nm1):
    cdef int i,j
    for i from 0 <= i < Nm1.shape[0]:
        for j from 0 <= j < Nm1.shape[1]:
            gsl_matrix_set(m1,i, j, Nm1[i,j])



cdef gsl_mul_perm_matrix(gsl_permutation * p, gsl_matrix *m1):
    cdef int i
    cdef size_t *ai
    cdef gsl_vector * v1
    cdef gsl_matrix * m2
    m2 = gsl_matrix_alloc(m1.size1, m1.size2)
    cdef gsl_vector_view vw1
    ai = gsl_permutation_data(p)
    # add check
    for i from 0 <= i < m1.size1:
        vw1 = gsl_matrix_row(m1, ai[i])
        v1 = &vw1.vector
        gsl_matrix_set_row(m2, i, v1)
    gsl_matrix_memcpy(m1, m2)
    gsl_matrix_free(m2)


cdef gsl_matrix_upperD(gsl_matrix *m2, gsl_matrix * m1):
    gsl_matrix_memcpy(m2,m1)
    cdef int i, j
    # add check
    for i from 0 <= i < m1.size1:
        for j from 0 <= j < i:
            gsl_matrix_set(m2,i,j,0.0)


cdef gsl_matrix_lowerU(gsl_matrix *m2, gsl_matrix * m1):
    # add check
    gsl_matrix_set_identity(m2)
    cdef int i, j
    for i from 0 <= i < m1.size1:
        for j from 0 <= j < i:
            gsl_matrix_set(m2,i,j, gsl_matrix_get(m1,i,j))

'''
factorize the square matrix A into the LU decomposition
  PA = LU
On output the diagonal and upper triangular part of the
input matrix A contain the matrix U. The lower triangular
part of the input matrix (excluding the diagonal) contains L.
The diagonal elements of L are unity, and are not stored.
'''

def t_gsl_linalg_LU_decomp():
    cdef gsl_matrix * m1, *m1c, *U, *L
    cdef gsl_permutation * p1
    p1 = gsl_permutation_alloc(2)
    cdef int sgn1
    m1 = gsl_matrix_alloc(2,2)
    m1c = gsl_matrix_alloc(2,2)
    U = gsl_matrix_alloc(2,2)
    L = gsl_matrix_alloc(2,2)
    Nm1 = array([[1,1],[2,5]])
    ary = []
    gsl_matrix_set_from_Num(m1, Nm1)
    gsl_matrix_memcpy(m1c, m1)
    gsl_linalg_LU_decomp(m1c, p1, &sgn1)
    # m1 = P*A
    gsl_mul_perm_matrix(p1, m1)
    # extract L and U
    gsl_matrix_upperD(U, m1c)
    gsl_matrix_lowerU(L, m1c)
    # U = L*U
    gsl_blas_dtrmm(CblasLeft, CblasLower, CblasNoTrans, CblasUnit, 1.0, L, U)
    # P*A = L*U
    gsl_matrix_sub(m1, U)
    for i in range(2):
        for j in range(2):
            ary.append(gsl_matrix_get(m1, i,j))
    ary.append(sgn1 + 1)
    gsl_matrix_free(m1); gsl_matrix_free(m1c)
    gsl_matrix_free(U); gsl_matrix_free(L)
    return ary

def t_gsl_linalg_LU_solve():
    cdef gsl_matrix * m1, *m1c
    cdef gsl_permutation * p1
    p1 = gsl_permutation_alloc(2)
    cdef int sgn1
    m1 = gsl_matrix_alloc(2,2)
    m1c = gsl_matrix_alloc(2,2)
    cdef gsl_vector * v1, * v2, * v3
    v1 = gsl_vector_alloc(2)
    v2 = gsl_vector_calloc(2)
    v3 = gsl_vector_alloc(2)
    gsl_vector_set_from_Num(v1, array([1.0,2.0]))
    gsl_matrix_set_from_Num(m1, array([[1,1],[2,5]]))
    ary = []
    gsl_matrix_memcpy(m1c, m1)
    gsl_linalg_LU_decomp(m1c, p1, &sgn1)
    # solve m1 * v3 = v1
    gsl_linalg_LU_solve(m1c, p1, v1, v3)
    # v2 = m1 * v3
    gsl_blas_dgemv(CblasNoTrans, 1.0, m1, v3, 1.0, v2)
    # v2 == v1
    for i in range(2):
        ary.append(gsl_vector_get(v2,i) - gsl_vector_get(v1,i))
    gsl_matrix_free(m1); gsl_matrix_free(m1c)
    gsl_vector_free(v1); gsl_vector_free(v2); gsl_vector_free(v3)
    return ary

def t_gsl_linalg_LU_svx():
    cdef gsl_matrix * m1, *m1c
    cdef gsl_permutation * p1
    p1 = gsl_permutation_alloc(2)
    cdef int sgn1
    m1 = gsl_matrix_alloc(2,2); m1c = gsl_matrix_alloc(2,2)
    cdef gsl_vector * v1, * v2, * v1c
    v1 = gsl_vector_alloc(2); v2 = gsl_vector_calloc(2)
    v1c = gsl_vector_alloc(2)
    gsl_vector_set_from_Num(v1, array([1.0,2.0]))
    gsl_matrix_set_from_Num(m1, array([[1,1],[2,5]]))
    ary = []
    gsl_matrix_memcpy(m1c, m1)
    gsl_vector_memcpy(v1c, v1)
    gsl_linalg_LU_decomp(m1c, p1, &sgn1)
    # solve m1 * x = v1c; x = v1c
    gsl_linalg_LU_svx(m1c, p1, v1c)
    # v2 = m1 * v1c
    gsl_blas_dgemv(CblasNoTrans, 1.0, m1, v1c, 1.0, v2)
    # v2 == v1
    for i in range(2):
        ary.append(gsl_vector_get(v2,i) - gsl_vector_get(v1,i))
    gsl_matrix_free(m1); gsl_matrix_free(m1c)
    gsl_vector_free(v1); gsl_vector_free(v2)
    return ary

def t_gsl_linalg_LU_refine():
    cdef int N
    N = 2
    cdef gsl_matrix * m1, *m1c
    cdef gsl_permutation * p1
    p1 = gsl_permutation_alloc(N)
    cdef int sgn1
    m1 = gsl_matrix_alloc(2,2); m1c = gsl_matrix_alloc(N,N)
    cdef gsl_vector * v1, * v2, * v1c, * res1
    v1 = gsl_vector_alloc(N); v2 = gsl_vector_calloc(N)
    v1c = gsl_vector_alloc(N); res1 = gsl_vector_calloc(N)
    gsl_vector_set_from_Num(v1, array([111111111111111.0,2.0]))
    gsl_matrix_set_from_Num(m1, array([[1,122222222.0],[2,5]]))
    ary = []
    gsl_matrix_memcpy(m1c, m1)
    gsl_vector_memcpy(v1c, v1)
    gsl_linalg_LU_decomp(m1c, p1, &sgn1)
    # solve m1 * x = v1c; x = v1c
    gsl_linalg_LU_svx(m1c, p1, v1c)
    gsl_linalg_LU_refine(m1, m1c, p1, v1, v1c, res1)
    # |res1| is > 1.0e-10
    ary.append((gsl_blas_dnrm2(res1) > 1.0e-10) - 1)
    # v2 = m1 * v1c
    gsl_blas_dgemv(CblasNoTrans, 1.0, m1, v1c, 1.0, v2)
    # v2 == v1 within precision 1e-15
    gsl_vector_sub(v2,v1)
    for i in range(2):
        ary.append(gsl_vector_get(v2,i))
    gsl_matrix_free(m1); gsl_matrix_free(m1c)
    gsl_vector_free(v1); gsl_vector_free(v2)
    return ary

def t_gsl_linalg_LU_invert():
    cdef gsl_matrix * m1, *m1c, *m1i, *m2
    cdef gsl_permutation * p1
    p1 = gsl_permutation_alloc(2)
    cdef int sgn1
    m1 = gsl_matrix_alloc(2,2); m1c = gsl_matrix_alloc(2,2)
    m1i = gsl_matrix_alloc(2,2); m2 = gsl_matrix_calloc(2,2)
    gsl_matrix_set_identity(m2)
    gsl_matrix_set_from_Num(m1, array([[1,1],[2,5]]))
    ary = []
    gsl_matrix_memcpy(m1c, m1)
    gsl_linalg_LU_decomp(m1c, p1, &sgn1)
    gsl_linalg_LU_invert(m1c, p1, m1i)
    # m1 * m1i == id
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, m1, m1i, -1.0, m2)
    for i in range(2):
        for j in range(2):
            ary.append(gsl_matrix_get(m2, i,j))
    gsl_matrix_free(m1); gsl_matrix_free(m1c)
    gsl_matrix_free(m1i); gsl_matrix_free(m2)
    return ary

def t_gsl_linalg_LU_det():
    cdef gsl_matrix * m1, *m1c
    cdef gsl_permutation * p1
    cdef double d
    p1 = gsl_permutation_alloc(2)
    cdef int sgn1
    m1 = gsl_matrix_alloc(2,2); m1c = gsl_matrix_alloc(2,2)
    gsl_matrix_set_from_Num(m1, array([[1,1],[2,5]]))
    ary = []
    gsl_matrix_memcpy(m1c, m1)
    gsl_linalg_LU_decomp(m1c, p1, &sgn1)
    d = gsl_linalg_LU_det(m1c, sgn1)
    ary.append(d - 3.0)
    d = gsl_linalg_LU_lndet(m1c)
    ary.append(d - log(3.0))
    ary.append(gsl_linalg_LU_sgndet(m1c, sgn1) - 1)
    gsl_matrix_free(m1); gsl_matrix_free(m1c)
    return ary


def t_gsl_linalg_householder_transform():
    cdef gsl_vector *v1, *v1c, *v2
    cdef double tau1, d2
    v1 = gsl_vector_alloc(3)
    v1c = gsl_vector_alloc(3)
    v2 = gsl_vector_alloc(3)
    gsl_vector_set_from_Num(v1, array([1,2,3]))
    d2 = gsl_blas_dnrm2(v1)
    gsl_vector_memcpy(v1c, v1)
    tau1 = gsl_linalg_householder_transform(v1c)
    gsl_linalg_householder_hv(tau1, v1c, v1)
    # v1 is v1c rotated so that only its first component is different from 0.
    ary = [fabs(gsl_vector_get(v1,0)) - d2]
    for i in range(1,v1.size):
        ary.append(gsl_vector_get(v1,i))
    gsl_vector_free(v1); gsl_vector_free(v1c); gsl_vector_free(v2)
    return ary

def t_gsl_linalg_householder_hm():
    cdef gsl_vector *v1, *v1c
    cdef gsl_matrix * m1
    m1 = gsl_matrix_alloc(3,3)
    cdef double tau1
    v1 = gsl_vector_alloc(3)
    v1c = gsl_vector_alloc(3)
    Nv1 = array([1,2,3])
    gsl_vector_set_from_Num(v1, Nv1)
    gsl_vector_memcpy(v1c, v1)
    tau1 = gsl_linalg_householder_transform(v1c)
    Nm1 = array([[1,2,4],[2,5,6],[3,8,9]])
    gsl_matrix_set_from_Num(m1, Nm1)
    # m1 = P * m1, where P = I - tau * v * v^T
    gsl_linalg_householder_hm(tau1, v1c, m1)
    ''' on the first column of m1 we can make the same check
        as in t_gsl_linalg_householder_transform()
    '''
    ary = [fabs(gsl_matrix_get(m1,0,0)) - gsl_blas_dnrm2(v1)]
    ary.extend([gsl_matrix_get(m1,1,0), gsl_matrix_get(m1,2,0)])
    gsl_matrix_free(m1)
    gsl_vector_free(v1); gsl_vector_free(v1c)
    return ary

def t_gsl_linalg_householder_mh():
    cdef gsl_vector *v1, *v1c
    cdef gsl_matrix * m1
    m1 = gsl_matrix_alloc(3,3)
    cdef double tau1
    v1 = gsl_vector_alloc(3)
    v1c = gsl_vector_alloc(3)
    Nv1 = array([1,2,3])
    gsl_vector_set_from_Num(v1, Nv1)
    gsl_vector_memcpy(v1c, v1)
    tau1 = gsl_linalg_householder_transform(v1c)
    Nm1 = array([[1,2,3],[2,5,6],[5,8,9]])
    gsl_matrix_set_from_Num(m1, Nm1)
    # m1 = P * m1
    gsl_linalg_householder_mh(tau1, v1c, m1)
    ''' on the first row of m1 we can make the same check
        as in t_gsl_linalg_householder_transform()
    '''
    ary = [fabs(gsl_matrix_get(m1,0,0)) - gsl_blas_dnrm2(v1)]
    ary.extend([gsl_matrix_get(m1,0,1), gsl_matrix_get(m1,0,2)])
    gsl_matrix_free(m1)
    gsl_vector_free(v1); gsl_vector_free(v1c)
    return ary

def t_gsl_linalg_QR_decomp():
    # see QR_decomp.py for an explanation
    cdef gsl_matrix *m1, *m1c, *qr
    cdef gsl_vector *tau
    tau = gsl_vector_alloc(2)
    m1 = gsl_matrix_alloc(3,2)
    m1c = gsl_matrix_alloc(3,2)
    qr = gsl_matrix_alloc(3,2)
    Nm1 = array([[1,1],[2,3],[2,1]])
    Nqr = array([[-3,-3],[0.5,-sqrt(2)],[0.5,1 - sqrt(2)]])
    gsl_matrix_set_from_Num(m1,Nm1)
    gsl_matrix_set_from_Num(qr,Nqr)
    gsl_matrix_memcpy(m1c,m1)
    gsl_linalg_QR_decomp(m1c, tau)
    ary = []
    for i from 0 <= i < 3:
        for j from 0 <= j < 2:
            ary.append(gsl_matrix_get(m1c, i,j) - Nqr[i,j])
    Nvh1 = array([1,0.5,0.5])
    ary.append(gsl_vector_get(tau,0) -2/dot(Nvh1,Nvh1))
    Nvh2 = array([1, 1-sqrt(2)])
    ary.append(gsl_vector_get(tau,1) -2/dot(Nvh2,Nvh2))
    gsl_matrix_free(m1); gsl_matrix_free(m1c);
    gsl_matrix_free(qr)
    gsl_vector_free(tau)
    return ary


def t_gsl_linalg_QR_solve():
    cdef gsl_matrix *m1
    cdef gsl_vector *tau, *b, *x
    tau = gsl_vector_alloc(3)
    x = gsl_vector_alloc(3)
    b = gsl_vector_alloc(3)
    m1 = gsl_matrix_alloc(3,3)
    Nm1 = array([[2,0,1],[6,2,0],[-3,-1,-1]])
    Nb = array([1,0,1])
    Nx = array([1, -3,-1])
    gsl_matrix_set_from_Num(m1,Nm1)
    gsl_vector_set_from_Num(b,Nb)
    gsl_linalg_QR_decomp(m1, tau)
    ary = []
    gsl_linalg_QR_solve(m1, tau, b, x)
    for i from 0 <= i < 3:
        ary.append(gsl_vector_get(x,i) - Nx[i])

    gsl_matrix_free(m1)
    gsl_vector_free(tau); gsl_vector_free(b); gsl_vector_free(x)
    return ary

def t_gsl_linalg_QR_svx():
    cdef gsl_matrix *m1
    cdef gsl_vector *tau, *b
    tau = gsl_vector_alloc(3)
    b = gsl_vector_alloc(3)
    m1 = gsl_matrix_alloc(3,3)
    Nm1 = array([[2,0,1],[6,2,0],[-3,-1,-1]])
    Nb = array([1,0,1])
    Nx = array([1, -3,-1])
    gsl_matrix_set_from_Num(m1,Nm1)
    gsl_vector_set_from_Num(b,Nb)
    gsl_linalg_QR_decomp(m1, tau)
    ary = []
    gsl_linalg_QR_svx(m1, tau, b)
    for i from 0 <= i < 3:
        ary.append(gsl_vector_get(b,i) - Nx[i])
    gsl_matrix_free(m1)
    gsl_vector_free(tau); gsl_vector_free(b)
    return ary

def t_gsl_linalg_QR_lssolve():
    cdef gsl_matrix *m1, *m1c
    cdef gsl_vector *tau, *res, *b, *x
    tau = gsl_vector_alloc(2)
    res = gsl_vector_alloc(3)
    b = gsl_vector_alloc(3)
    x = gsl_vector_alloc(2)
    m1 = gsl_matrix_alloc(3,2)
    m1c = gsl_matrix_alloc(3,2)
    Nm1 = array([[1,0],[0,1],[0,0]])
    Nb = array([1,0,3])
    gsl_matrix_set_from_Num(m1,Nm1)
    gsl_vector_set_from_Num(b, Nb)
    gsl_matrix_memcpy(m1c,m1)
    gsl_linalg_QR_decomp(m1c, tau)
    gsl_linalg_QR_lssolve(m1c, tau, b, x, res)
    ary = []
    for i in range(2):
        ary.append(gsl_vector_get(res,i))
    ary.append(gsl_vector_get(res,2) - 3)
    gsl_matrix_free(m1); gsl_matrix_free(m1c)
    gsl_vector_free(tau); gsl_vector_free(res)
    gsl_vector_free(b); gsl_vector_free(x)
    return ary

def t_gsl_linalg_QR_Qvec():
    # see QR_decomp.py for an explanation
    cdef gsl_matrix *m1, *m1c
    cdef gsl_vector *tau, *v1, *v1c
    tau = gsl_vector_alloc(2)
    v1 = gsl_vector_alloc(3)
    v1c = gsl_vector_alloc(3)
    m1 = gsl_matrix_alloc(3,2)
    m1c = gsl_matrix_alloc(3,2)
    Nm1 = array([[1,1],[2,3],[2,1]])
    Nv1 = array([1,2,3])
    gsl_vector_set_from_Num(v1, Nv1)
    gsl_vector_memcpy(v1c, v1)
    gsl_matrix_set_from_Num(m1,Nm1)
    gsl_matrix_memcpy(m1c,m1)
    gsl_linalg_QR_decomp(m1c, tau)
    ary = []
    gsl_linalg_QR_Qvec(m1c, tau, v1c)
    NQ = array([[-1.0/3,     0,      -2*sqrt(2)/3],[-2.0/3,   -sqrt(2)/2,sqrt(2)/6],[-2.0/3,   sqrt(2)/2,sqrt(2)/6]])
    Nv2 = dot(NQ,Nv1)
    for i in range(3):
        ary.append(gsl_vector_get(v1c,i) - Nv2[i])
    gsl_vector_memcpy(v1c, v1)
    gsl_linalg_QR_QTvec(m1c, tau, v1c)
    Nv2 = dot(transpose(NQ),Nv1)
    for i in range(3):
        ary.append(gsl_vector_get(v1c,i) - Nv2[i])
    cdef gsl_matrix *q1, *r1
    q1 = gsl_matrix_alloc(3,3)
    r1 = gsl_matrix_alloc(3,2)

    # gsl_linalg_QR_unpack
    gsl_linalg_QR_unpack(m1c, tau, q1, r1)
    for i in range(3):
        for j in range(3):
            ary.append(gsl_matrix_get(q1,i,j) - NQ[i,j])
    NR = array([[-3,-3],[0,-sqrt(2)],[0,0]])
    for i in range(3):
        for j in range(2):
            ary.append(gsl_matrix_get(r1,i,j) - NR[i,j])

    gsl_matrix_free(m1); gsl_matrix_free(m1c)
    gsl_matrix_free(q1); gsl_matrix_free(r1)
    gsl_vector_free(tau)
    gsl_vector_free(v1); gsl_vector_free(v1c);
    return ary


def t_gsl_linalg_R_solve():
    cdef gsl_matrix *m1
    cdef gsl_vector *x, *b
    m1 = gsl_matrix_alloc(2,2)
    x = gsl_vector_alloc(2)
    b = gsl_vector_alloc(2)
    Nm1 = array([[1,2],[0,3]])
    Nx = array([1,2])
    Nb = array([5, 6])
    gsl_matrix_set_from_Num(m1, Nm1)
    gsl_vector_set_from_Num(b,Nb)
    gsl_linalg_R_solve(m1, b, x)
    ary = []
    for i in range(2):
        ary.append(gsl_vector_get(x,i) - Nx[i])
    '''
    gsl_linalg_R_svx  missing
    gsl_linalg_R_svx(m1, b)
    for i in range(2):
      ary.append(gsl_vector_get(b,i) - Nx[i])
    '''
    return ary


def t_gsl_linalg_QR_QRsolve():
    cdef gsl_matrix *m1, *qr, *q, *r
    cdef gsl_vector *x, *b, *tau, *b1
    m1 = gsl_matrix_alloc(2,2)
    qr = gsl_matrix_alloc(2,2)
    q = gsl_matrix_alloc(2,2)
    r = gsl_matrix_alloc(2,2)
    x = gsl_vector_alloc(2)
    b = gsl_vector_alloc(2)
    b1 = gsl_vector_calloc(2)
    tau = gsl_vector_alloc(2)
    Nm1 = array([[1,2],[1,3]])
    Nx = array([1,2])
    Nb = array([5, 7])
    gsl_matrix_set_from_Num(m1, Nm1)
    gsl_matrix_memcpy(qr, m1)
    gsl_vector_set_from_Num(b,Nb)
    gsl_linalg_QR_decomp(qr,tau)
    gsl_linalg_QR_unpack(qr, tau, q,r)
    gsl_linalg_QR_QRsolve(q,r,b,x)
    ary = []
    for i in range(2):
        ary.append(gsl_vector_get(x,i) - Nx[i])

    # gsl_linalg_QR_Rsolve
    gsl_vector_set_zero(x)
    gsl_blas_dgemv(CblasTrans, 1.0, q, b, 0, b1)
    gsl_linalg_QR_Rsolve(qr, b1,x)
    for i in range(2):
        ary.append(gsl_vector_get(x,i) - Nx[i])

    # gsl_linalg_QR_Rsvx
    gsl_vector_memcpy(x,b1)
    gsl_linalg_QR_Rsvx(qr, x)
    for i in range(2):
        ary.append(gsl_vector_get(x,i) - Nx[i])

    gsl_matrix_free(m1)
    gsl_matrix_free(qr)
    gsl_matrix_free(q)
    gsl_matrix_free(r)
    gsl_vector_free(x)
    gsl_vector_free(b)
    gsl_vector_free(b1)
    gsl_vector_free(tau)
    return ary

def t_gsl_linalg_SV_decomp():
    cdef gsl_matrix *a, *v, *m1
    cdef gsl_vector *s, *work
    a = gsl_matrix_alloc(3,2)
    m1 = gsl_matrix_alloc(3,2)
    v = gsl_matrix_alloc(2,2)
    s = gsl_vector_alloc(2)
    work = gsl_vector_alloc(2)
    Na = array([[1,1],[0,1],[1,0]])
    gsl_matrix_set_from_Num(a,Na)
    gsl_matrix_memcpy(m1,a)
    gsl_linalg_SV_decomp(a, v, s, work)
    Nu = array([[-sqrt(6)/3, 0], [-sqrt(6)/6, -sqrt(2)/2], [-sqrt(6)/6, sqrt(2)/2]])
    Ns = array([[sqrt(3),0],[0,1]])
    Nv = array([[-1,1],[-1,-1]])*sqrt(2)/2
    Na1 = dot(Nu, dot(Ns,transpose(Nv)))
    ary = []
    for i in range(3):
        for j in range(2):
            ary.append(Na1[i,j] - Na[i,j])
            ary.append(gsl_matrix_get(a,i,j) - Nu[i,j])
    for i in range(2):
        ary.append(gsl_vector_get(s,i) - Ns[i,i])
        for j in range(2):
            ary.append(gsl_matrix_get(v,i,j) - Nv[i,j])

    # gsl_linalg_SV_solve
    cdef gsl_vector *b, *x, *tau, *x1, *res
    b = gsl_vector_alloc(3)
    tau = gsl_vector_alloc(2)
    res = gsl_vector_alloc(3)
    x = gsl_vector_alloc(2)
    x1 = gsl_vector_alloc(2)
    Nb = array([3,4,5])
    gsl_vector_set_from_Num(b,Nb)
    gsl_linalg_QR_decomp(m1, tau)
    gsl_linalg_QR_lssolve(m1, tau, b, x1, res)
    '''
    x is the least squares solution to the overdetermined
    system a x = b
    '''
    gsl_linalg_SV_solve(a,v,s,b,x)
    for i in range(2):
        ary.append(gsl_vector_get(x,i) - gsl_vector_get(x1,i))

    gsl_matrix_free(a)
    gsl_matrix_free(v)
    gsl_vector_free(s)
    gsl_vector_free(work)
    return ary

def t_gsl_linalg_SV_decomp_mod():
    cdef gsl_matrix *a, *v, *X
    cdef gsl_vector *s, *work
    a = gsl_matrix_alloc(3,2)
    v = gsl_matrix_alloc(2,2)
    X = gsl_matrix_alloc(2,2)
    s = gsl_vector_alloc(2)
    work = gsl_vector_alloc(2)
    Na = array([[1,1],[0,1],[1,0]])
    gsl_matrix_set_from_Num(a,Na)
    gsl_linalg_SV_decomp_mod(a, X, v, s, work)
    Nu = array([[-sqrt(6)/3, 0], [-sqrt(6)/6, -sqrt(2)/2], [-sqrt(6)/6, sqrt(2)/2]])
    Ns = array([[sqrt(3),0],[0,1]])
    Nv = array([[-1,1],[-1,-1]])*sqrt(2)/2
    Na1 = dot(Nu, dot(Ns,transpose(Nv)))
    ary = []
    for i in range(3):
        for j in range(2):
            ary.append(Na1[i,j] - Na[i,j])
            ary.append(gsl_matrix_get(a,i,j) - Nu[i,j])
    for i in range(2):
        ary.append(gsl_vector_get(s,i) - Ns[i,i])
        for j in range(2):
            ary.append(gsl_matrix_get(v,i,j) - Nv[i,j])
    gsl_matrix_free(a)
    gsl_matrix_free(v)
    gsl_matrix_free(X)
    gsl_vector_free(s)
    gsl_vector_free(work)
    return ary

def t_gsl_linalg_SV_decomp_jacobi():
    cdef gsl_matrix *a, *v
    cdef gsl_vector *s
    a = gsl_matrix_alloc(3,2)
    v = gsl_matrix_alloc(2,2)
    s = gsl_vector_alloc(2)
    Na = array([[1,1],[0,1],[1,0]])
    gsl_matrix_set_from_Num(a,Na)
    gsl_linalg_SV_decomp_jacobi(a, v, s)
    Nu = array([[-sqrt(6)/3, 0], [-sqrt(6)/6, -sqrt(2)/2], [-sqrt(6)/6, sqrt(2)/2]])
    Ns = array([[sqrt(3),0],[0,1]])
    Nv = array([[-1,1],[-1,-1]])*sqrt(2)/2
    Na1 = dot(Nu, dot(Ns,transpose(Nv)))
    ary = []
    # opposite signs for U and V wrt previous tests
    for i in range(3):
        for j in range(2):
            ary.append(Na1[i,j] - Na[i,j])
            ary.append(gsl_matrix_get(a,i,j) + Nu[i,j])
    for i in range(2):
        ary.append(gsl_vector_get(s,i) - Ns[i,i])
    for i in range(2):
        for j in range(2):
            ary.append(gsl_matrix_get(v,i,j) + Nv[i,j])

    gsl_matrix_free(a)
    gsl_matrix_free(v)
    gsl_vector_free(s)
    return ary


def t_gsl_linalg_cholesky_decomp():
    cdef gsl_matrix *m
    m = gsl_matrix_alloc(2,2)
    Nm = array([[2, -2], [-2, 5]])
    NL = array([[sqrt(2), 0],[-sqrt(2), sqrt(3)]])
    Nm1 = dot(NL, transpose(NL))
    gsl_matrix_set_from_Num(m, Nm)
    ary = []
    for i in range(2):
        for j in range(2):
            ary.append(gsl_matrix_get(m,i,j) - Nm1[i,j])

    gsl_linalg_cholesky_decomp(m)
    for i in range(2):
        for j in range(0,i+1):
            ary.append(gsl_matrix_get(m,i,j) - NL[i,j])
    gsl_matrix_free(m)
    return ary

def t_gsl_linalg_cholesky_solve():
    cdef gsl_matrix *m
    cdef gsl_vector *b, *x
    b = gsl_vector_alloc(2)
    x = gsl_vector_alloc(2)
    m = gsl_matrix_alloc(2,2)
    Nm = array([[2, -2], [-2, 5]])
    Nx = array([1,1])
    Nb = array([0,3])
    gsl_vector_set_from_Num(b,Nb)
    gsl_matrix_set_from_Num(m, Nm)
    gsl_linalg_cholesky_decomp(m)
    ary = []
    gsl_linalg_cholesky_solve(m, b, x)
    for i in range(2):
        ary.append(gsl_vector_get(x,i) - Nx[i])

    # gsl_linalg_cholesky_svx
    gsl_linalg_cholesky_svx(m,b)
    for i in range(2):
        ary.append(gsl_vector_get(b,i) - Nx[i])
    gsl_matrix_free(m)
    gsl_vector_free(b)
    gsl_vector_free(x)
    return ary

def t_gsl_linalg_HH_solve():
    cdef gsl_matrix *a, *a1
    cdef gsl_vector *x, *b
    a = gsl_matrix_alloc(3,3)
    a1 = gsl_matrix_alloc(3,3)
    x = gsl_vector_alloc(3)
    b = gsl_vector_alloc(3)
    Na = array([[1,3,4], [3,2,6], [2,8,3]])
    Nx = array([1,2,3])
    Nb = array([19, 25, 27])
    gsl_matrix_set_from_Num(a, Na)
    gsl_matrix_memcpy(a1,a)
    gsl_vector_set_from_Num(b, Nb)
    gsl_linalg_HH_solve(a,b,x)
    ary = []
    for i in range(3):
        ary.append(gsl_vector_get(x,i) - Nx[i])

    # gsl_linalg_HH_svx
    gsl_linalg_HH_svx(a1,b)
    for i in range(3):
        ary.append(gsl_vector_get(b,i) - Nx[i])
    gsl_matrix_free(a)
    gsl_matrix_free(a1)
    gsl_vector_free(x)
    gsl_vector_free(b)
    return ary

def t_gsl_linalg_solve_symm_tridiag():
    cdef gsl_matrix *a
    a = gsl_matrix_alloc(4,4)
    cdef gsl_vector *diag, *e, *b, *x
    diag = gsl_vector_alloc(4)
    b = gsl_vector_alloc(4)
    x = gsl_vector_alloc(4)
    e = gsl_vector_alloc(3)
    Ndiag = array([1,2,3,4])
    Ne = array([1,3,5])
    Na = array([[1,1,0,0], [1,2,3,0], [0,3,3,5], [0,0,5,4]])
    Nx = array([0,1,0,0])
    gsl_vector_set_from_Num(diag,Ndiag)
    gsl_vector_set_from_Num(e, Ne)
    # A*x = b
    Nb = array([1, 2, 3, 0])
    gsl_vector_set_from_Num(b,Nb)
    gsl_linalg_solve_symm_tridiag(diag, e, b,x)
    ary = []
    for i in range(4):
        ary.append(gsl_vector_get(x,i) - Nx[i])
    gsl_matrix_free(a)
    gsl_vector_free(diag); gsl_vector_free(e)
    gsl_vector_free(b); gsl_vector_free(x)
    return ary

def t_gsl_linalg_solve_symm_cyc_tridiag():
    cdef gsl_matrix *a
    a = gsl_matrix_alloc(4,4)
    cdef gsl_vector *diag, *e, *b, *x
    diag = gsl_vector_alloc(4)
    b = gsl_vector_alloc(4)
    x = gsl_vector_alloc(4)
    e = gsl_vector_alloc(4)
    Ndiag = array([1,2,3,4])
    Ne = array([1,3,5,7])
    Na = array([[1,1,0,7], [1,2,3,0], [0,3,3,5], [7,0,5,4]])
    Nx = array([0,0,0,1])
    gsl_vector_set_from_Num(diag,Ndiag)
    gsl_vector_set_from_Num(e, Ne)
    # A*x = b
    Nb = array([7, 0, 5, 4])
    gsl_vector_set_from_Num(b,Nb)
    gsl_linalg_solve_symm_cyc_tridiag(diag, e, b,x)
    ary = []
    for i in range(4):
        ary.append(gsl_vector_get(x,i) - Nx[i])
    gsl_matrix_free(a)
    gsl_vector_free(diag); gsl_vector_free(e)
    gsl_vector_free(b); gsl_vector_free(x)
    return ary

def t_gsl_linalg_symmtd_decomp():
    cdef gsl_matrix *a, *q, *c, *t
    cdef gsl_vector *tau, *diag, * subdiag,
    cdef gsl_vector_view wdiag, wsubdiag
    a = gsl_matrix_alloc(4,4)
    q = gsl_matrix_alloc(4,4)
    t = gsl_matrix_calloc(4,4)
    c = gsl_matrix_calloc(4,4)
    tau = gsl_vector_alloc(3)
    diag = gsl_vector_alloc(4)
    subdiag = gsl_vector_alloc(3)
    Na = array([[1,1,0,7], [1,2,3,0], [0,3,3,5], [7,0,5,4]])
    gsl_matrix_set_from_Num(a, Na)
    gsl_linalg_symmtd_decomp(a, tau)
    gsl_linalg_symmtd_unpack(a, tau, q, diag, subdiag)
    wdiag = gsl_matrix_diagonal(t)
    wsubdiag = gsl_matrix_subdiagonal(t,1)
    gsl_vector_memcpy(&wdiag.vector, diag)
    gsl_vector_memcpy(&wsubdiag.vector, subdiag)
    wsubdiag = gsl_matrix_superdiagonal(t,1)
    gsl_vector_memcpy(&wsubdiag.vector, subdiag)
    # T*Q^T
    gsl_blas_dgemm(CblasNoTrans, CblasTrans,1.0,t,q,0.0,c)
    # Q*c = Q*T*Q^T
    cdef gsl_matrix *c1
    c1 = gsl_matrix_calloc(4,4)
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0,q,c,0.0, c1)
    ary = []
    # A == Q*T*Q^T
    for i in range(4):
        for j in range(4):
            ary.append(gsl_matrix_get(c1,i,j) - Na[i,j])

    # gsl_linalg_symmtd_unpack_T
    cdef gsl_matrix *a1
    cdef gsl_vector *diag1, * subdiag1
    a1 = gsl_matrix_alloc(4,4)
    gsl_matrix_set_from_Num(a1, Na)
    diag1 = gsl_vector_alloc(4)
    subdiag1 = gsl_vector_alloc(3)
    gsl_linalg_symmtd_decomp(a1, tau)
    gsl_linalg_symmtd_unpack_T(a1, diag1, subdiag1)
    for i in range(4):
        ary.append(gsl_vector_get(diag1,i) - gsl_vector_get(diag,i))
    for i in range(3):
        ary.append(gsl_vector_get(subdiag1,i) - gsl_vector_get(subdiag,i))
    gsl_matrix_free(a); gsl_matrix_free(q); gsl_matrix_free(c)
    gsl_matrix_free(t); gsl_matrix_free(c1)
    gsl_vector_free(tau); gsl_vector_free(diag); gsl_vector_free(subdiag)
    gsl_matrix_free(a1)
    gsl_vector_free(diag1); gsl_vector_free(subdiag1)
    return ary

def t_gsl_linalg_bidiag_decomp():
    cdef gsl_matrix *a, *u, *v, *b
    cdef gsl_vector *tau_U, *tau_V, *diag, *superdiag
    cdef gsl_vector_view wdiag, wsuperdiag
    a = gsl_matrix_alloc(4,3)
    b = gsl_matrix_calloc(3,3)
    u = gsl_matrix_alloc(4,3)
    v = gsl_matrix_alloc(3,3)
    tau_U = gsl_vector_alloc(3)
    tau_V = gsl_vector_alloc(2)
    diag = gsl_vector_alloc(3)
    superdiag = gsl_vector_alloc(2)
    Na = array([[1,3,4], [3,2,8], [4,8,3], [5,9,7]])
    gsl_matrix_set_from_Num(a, Na)
    gsl_linalg_bidiag_decomp(a, tau_U, tau_V)
    gsl_linalg_bidiag_unpack(a, tau_U, u, tau_V, v, diag, superdiag)
    wdiag = gsl_matrix_diagonal(b)
    gsl_vector_memcpy(&wdiag.vector, diag)
    wsuperdiag = gsl_matrix_superdiagonal(b,1)
    gsl_vector_memcpy(&wsuperdiag.vector, superdiag)
    # B * V^T
    cdef gsl_matrix *c33
    c33 = gsl_matrix_alloc(3,3)
    gsl_blas_dgemm(CblasNoTrans, CblasTrans,1.0,b,v,0.0,c33)
    # U * c33 = U * B * V^T
    cdef gsl_matrix *c43
    c43 = gsl_matrix_alloc(4,3)
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0,u,c33,0.0, c43)
    ary = []
    for i in range(4):
        for j in range(3):
            ary.append(gsl_matrix_get(c43,i,j) - Na[i,j])
    gsl_matrix_free(a); gsl_matrix_free(u); gsl_matrix_free(v); gsl_matrix_free(b)
    gsl_matrix_free(c33); gsl_matrix_free(c43)
    gsl_vector_free(tau_U); gsl_vector_free(tau_V)
    gsl_vector_free(diag); gsl_vector_free(superdiag)
    return ary
