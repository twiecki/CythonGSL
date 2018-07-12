from cython_gsl cimport *
from libc.stdlib cimport *

cdef extern  from "gsl/gsl_spmatrix.h":
	ctypedef struct gsl_spmatrix_tree:
		void *tree
		void *node_array
		size_t n

	ctypedef struct gsl_spmatrix:
		size_t size1
		size_t size2
		size_t *i
		double *data
		size_t *p
		size_t nzmax
		size_t nz
		gsl_spmatrix_tree *tree_data
		void *work
		size_t sptype

	size_t GSL_SPMATRIX_TRIPLET
	size_t GSL_SPMATRIX_CCS
	size_t GSL_SPMATRIX_CRS

	int GSL_SPMATRIX_ISTRIPLET(size_t *m) nogil
	int GSL_SPMATRIX_ISCCS(size_t *m) nogil
	int GSL_SPMATRIX_ISCRS(size_t *m) nogil

	gsl_spmatrix *gsl_spmatrix_alloc(const size_t n1, const size_t n2) nogil

	gsl_spmatrix *gsl_spmatrix_alloc_nzmax(const size_t n1, const size_t n2, const size_t nzmax, const size_t flags)  nogil
	
	void gsl_spmatrix_free(gsl_spmatrix *m) nogil

	int gsl_spmatrix_realloc(const size_t nzmax, gsl_spmatrix *m) nogil

	int gsl_spmatrix_set_zero(gsl_spmatrix *m) nogil

	size_t gsl_spmatrix_nnz(const gsl_spmatrix *m) nogil

	int gsl_spmatrix_compare_idx(const size_t ia, const size_t ja, const size_t ib, const size_t jb) nogil

	int gsl_spmatrix_tree_rebuild(gsl_spmatrix * m) nogil

	# /* spcopy.c */
	int gsl_spmatrix_memcpy(gsl_spmatrix *dest, const gsl_spmatrix *src)  nogil

	# /* spgetset.c */
	double gsl_spmatrix_get(const gsl_spmatrix *m, const size_t i, const size_t j) nogil

	int gsl_spmatrix_set(gsl_spmatrix *m, const size_t i, const size_t j, const double x) nogil

	double *gsl_spmatrix_ptr(gsl_spmatrix *m, const size_t i, const size_t j) nogil

	# /* spcompress.c */
	gsl_spmatrix *gsl_spmatrix_compcol(const gsl_spmatrix *T) nogil

	gsl_spmatrix *gsl_spmatrix_ccs(const gsl_spmatrix *T) nogil

	gsl_spmatrix *gsl_spmatrix_crs(const gsl_spmatrix *T) nogil

	void gsl_spmatrix_cumsum(const size_t n, size_t *c) nogil

	# /* spio.c */
	int gsl_spmatrix_fprintf(FILE *stream, const gsl_spmatrix *m, const char *format) nogil

	gsl_spmatrix * gsl_spmatrix_fscanf(FILE *stream) nogil

	int gsl_spmatrix_fwrite(FILE *stream, const gsl_spmatrix *m) nogil

	int gsl_spmatrix_fread(FILE *stream, gsl_spmatrix *m) nogil

	# /* spoper.c */
	int gsl_spmatrix_scale(gsl_spmatrix *m, const double x) nogil

	int gsl_spmatrix_minmax(const gsl_spmatrix *m, double *min_out, double *max_out) nogil

	int gsl_spmatrix_add(gsl_spmatrix *c, const gsl_spmatrix *a, const gsl_spmatrix *b) nogil

	int gsl_spmatrix_d2sp(gsl_spmatrix *S, const gsl_matrix *A) nogil

	int gsl_spmatrix_sp2d(gsl_matrix *A, const gsl_spmatrix *S) nogil

	# /* spprop.c */
	int gsl_spmatrix_equal(const gsl_spmatrix *a, const gsl_spmatrix *b) nogil

	# /* spswap.c */
	int gsl_spmatrix_transpose(gsl_spmatrix * m) nogil

	int gsl_spmatrix_transpose2(gsl_spmatrix * m) nogil

	int gsl_spmatrix_transpose_memcpy(gsl_spmatrix *dest, const gsl_spmatrix *src) nogil
