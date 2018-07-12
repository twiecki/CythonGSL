from cython_gsl cimport *

cdef extern  from "gsl/gsl_splinalg.h":
	ctypedef struct gsl_splinalg_itersolve_type:
		const char *name
		void * (*alloc) (const size_t n, const size_t m)
		int (*iterate) (const gsl_spmatrix *A, const gsl_vector *b, const double tol, gsl_vector *x, void *)
		double (*normr)(const void *)
		void (*free) (void *)

	ctypedef struct gsl_splinalg_itersolve:
		const gsl_splinalg_itersolve_type * type
		double normr
		void * state

	gsl_splinalg_itersolve_type * gsl_splinalg_itersolve_gmres

	gsl_splinalg_itersolve * gsl_splinalg_itersolve_alloc(const gsl_splinalg_itersolve_type *T, const size_t n, const size_t m) nogil

	void gsl_splinalg_itersolve_free(gsl_splinalg_itersolve *w) nogil

	const char *gsl_splinalg_itersolve_name(const gsl_splinalg_itersolve *w) nogil

	int gsl_splinalg_itersolve_iterate(const gsl_spmatrix *A, const gsl_vector *b, const double tol, gsl_vector *x, gsl_splinalg_itersolve *w) nogil

	double gsl_splinalg_itersolve_normr(const gsl_splinalg_itersolve *w) nogil
