from cython_gsl cimport *

cdef extern  from "gsl/gsl_spblas.h":
	int gsl_spblas_dgemv(const CBLAS_TRANSPOSE_t TransA, const double alpha,
                     const gsl_spmatrix *A, const gsl_vector *x,
                     const double beta, gsl_vector *y) nogil

	int gsl_spblas_dgemm(const double alpha, const gsl_spmatrix *A,
	                     const gsl_spmatrix *B, gsl_spmatrix *C) nogil

	size_t gsl_spblas_scatter(const gsl_spmatrix *A, const size_t j,
	                          const double alpha, size_t *w, double *x,
	                          const size_t mark, gsl_spmatrix *C, size_t nz) nogil
