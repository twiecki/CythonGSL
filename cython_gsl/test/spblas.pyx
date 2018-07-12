from cython_gsl cimport *
cimport numpy as np
import numpy as np
from scipy.sparse import csc_matrix

def t_gsl_spblas_dgemm():
	# A = [[ 1, 0, 2 ],
	# 	   [ 0, 3, 0]]
	cdef size_t a_m = 2
	cdef size_t a_n = 3
	cdef size_t a_nnz = 3
	cdef double[3] a
	a[0] = 1.0
	a[1] = 3.0
	a[2] = 2.0
	cdef size_t[4] ja
	ja[0] = 0
	ja[1] = 1
	ja[2] = 2
	ja[3] = 3
	cdef size_t[3] ia
	ia[0] = 0
	ia[1] = 1
	ia[2] = 0
	cdef gsl_spmatrix *A = gsl_spmatrix_alloc_nzmax(a_m, a_n, a_nnz, GSL_SPMATRIX_CCS)
	A.i = &ia[0]
	A.p = &ja[0]
	A.data = &a[0]
	A.nz = a_nnz
	A.nzmax = a_nnz

	# B = [[0, 0, 0, 4],
	# 	   [0, 0, 5, 6],
	# 	   [0, 0, 0, 0]]
	cdef size_t b_m = 3
	cdef size_t b_n = 4
	cdef size_t b_nnz = 3
	cdef double[3] b
	b[0] = 5.0
	b[1] = 4.0
	b[2] = 6.0
	cdef size_t[5] jb
	jb[0] = 0
	jb[1] = 0
	jb[2] = 0
	jb[3] = 1
	jb[4] = 3
	cdef size_t[3] ib
	ib[0] = 1
	ib[1] = 0
	ib[2] = 1
	cdef gsl_spmatrix *B = gsl_spmatrix_alloc_nzmax(b_m, b_n, b_nnz, GSL_SPMATRIX_CCS)
	B.i = &ib[0]
	B.p = &jb[0]
	B.data = &b[0]
	B.nz = b_nnz
	B.nzmax = b_nnz

	# Expected : C = A * B
	# C = [[0, 0, 0,  4],
	# 	   [0, 0, 15, 18]]
	cdef gsl_spmatrix * C = gsl_spmatrix_alloc_nzmax(a_m, b_n, 3, GSL_SPMATRIX_CCS)
	gsl_spblas_dgemm(1.0, A, B, C)

	c_nnz = C.nz
	c = np.asarray(<double[:c_nnz]> C.data)
	ic = np.asarray(<size_t[:c_nnz]> C.i)
	jc = np.asarray(<size_t[:b_n+1]> C.p)
	C_dense = csc_matrix((c, ic, jc), shape=(a_m, b_n)).todense().tolist()
	return C_dense

def t_gsl_spblas_dgemv():
	# A = [[ 1, 0, 2 ],
	# 	   [ 0, 3, 0]]
	cdef size_t a_m = 2
	cdef size_t a_n = 3
	cdef size_t a_nnz = 3
	cdef double[3] a
	a[0] = 1.0
	a[1] = 3.0
	a[2] = 2.0
	cdef size_t[4] ja
	ja[0] = 0
	ja[1] = 1
	ja[2] = 2
	ja[3] = 3
	cdef size_t[3] ia
	ia[0] = 0
	ia[1] = 1
	ia[2] = 0
	cdef gsl_spmatrix *A = gsl_spmatrix_alloc_nzmax(a_m, a_n, a_nnz, GSL_SPMATRIX_CCS)
	A.i = &ia[0]
	A.p = &ja[0]
	A.data = &a[0]
	A.nz = a_nnz
	A.nzmax = a_nnz

	# x = [4, 5, 6]
	cdef gsl_vector * x = gsl_vector_calloc(a_n)
	x.data[0] = 4.0
	x.data[1] = 5.0
	x.data[2] = 6.0
	
	# Expected : y = A * x
	# y = [16, 15]
	cdef gsl_vector * y = gsl_vector_calloc(a_m)
	gsl_spblas_dgemv(CblasNoTrans, 1.0, A, x, 1.0, y)

	return np.asarray(<double[:y.size]> y.data).tolist()
