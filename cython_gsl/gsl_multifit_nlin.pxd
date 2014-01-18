from cython_gsl cimport *

cdef extern from "gsl/gsl_multifit_nlin.h":

    int gsl_multifit_gradient (const gsl_matrix * J,
                               const gsl_vector * f,
                               gsl_vector * g) nogil

    int gsl_multifit_covar (const gsl_matrix * J,
                            double epsrel,
                            gsl_matrix * covar) nogil

    # Definition of vector-valued functions with parameters based on
    # gsl_vector
    ctypedef struct gsl_multifit_function:
        int (* f) (const gsl_vector * x, void * params, gsl_vector * f) nogil
        size_t n
        size_t p
        void * params

    ctypedef struct gsl_multifit_fsolver_type:
        const char * name
        size_t size
        int (* alloc) (void * state, size_t n, size_t p) nogil
        int (* set) (void * state, gsl_multifit_function * function,
                     gsl_vector * x, gsl_vector * f, gsl_vector * dx) nogil
        int (* iterate) (void * state, gsl_multifit_function * function,
                         gsl_vector * x, gsl_vector * f, gsl_vector * dx) nogil
        void (* free) (void * state)

    ctypedef struct gsl_multifit_fsolver:
        const gsl_multifit_fsolver_type * type
        gsl_multifit_function * function
        gsl_vector * x
        gsl_vector * f
        gsl_vector * dx
        void * state

    gsl_multifit_fsolver * gsl_multifit_fsolver_alloc (const gsl_multifit_fsolver_type * T,
                                                       size_t n, size_t p) nogil

    void gsl_multifit_fsolver_free (gsl_multifit_fsolver * s) nogil

    int gsl_multifit_fsolver_set (gsl_multifit_fsolver * s, gsl_multifit_function * f,
                                  const gsl_vector * x) nogil

    int gsl_multifit_fsolver_iterate (gsl_multifit_fsolver * s) nogil

    int gsl_multifit_fsolver_driver (gsl_multifit_fsolver * s, const size_t maxiter,
                                     const double epsabs, const double epsrel) nogil

    # Definition of vector-valued functions and gradient with parameters
    # based on gsl_vector
    ctypedef struct gsl_multifit_function_fdf:
        int (* f) (const gsl_vector * x, void * params, gsl_vector * f) nogil
        int (* df) (const gsl_vector * x, void * params, gsl_matrix * df) nogil
        int (* fdf) (const gsl_vector * x, void * params, gsl_vector * f,
                     gsl_matrix * df) nogil
        size_t n
        size_t p
        void * params

    ctypedef struct gsl_multifit_fdfsolver_type:
        const char * name
        size_t size
        int (* alloc) (void * state, size_t n, size_t p) nogil
        int (* set) (void * state, gsl_multifit_function_fdf * fdf, gsl_vector * x,
                     gsl_vector * f, gsl_matrix * J, gsl_vector * dx) nogil
        int (* iterate) (void * state, gsl_multifit_function_fdf * fdf,
                         gsl_vector * x, gsl_vector * f, gsl_matrix * J,
                         gsl_vector * dx) nogil
        void (* free) (void * state)

    ctypedef struct gsl_multifit_fdfsolver:
        const gsl_multifit_fdfsolver_type * type
        gsl_multifit_function_fdf * fdf
        gsl_vector * x
        gsl_vector * f
        gsl_matrix * J
        gsl_vector * dx
        void * state

    gsl_multifit_fdfsolver * gsl_multifit_fdfsolver_alloc (const gsl_multifit_fdfsolver_type * T,
                                                           size_t n, size_t p) nogil

    int gsl_multifit_fdfsolver_set (gsl_multifit_fdfsolver * s,
                                    gsl_multifit_function_fdf * fdf,
                                    const gsl_vector * x) nogil

    int gsl_multifit_fdfsolver_iterate (gsl_multifit_fdfsolver * s) nogil

    int gsl_multifit_fdfsolver_driver (gsl_multifit_fdfsolver * s, const size_t maxiter,
                                       const double epsabs, const double epsrel) nogil

    void gsl_multifit_fdfsolver_free (gsl_multifit_fdfsolver * s) nogil

    const char * gsl_multifit_fdfsolver_name (const gsl_multifit_fdfsolver * s) nogil

    gsl_vector * gsl_multifit_fdfsolver_position (const gsl_multifit_fdfsolver * s) nogil

    int gsl_multifit_test_delta (const gsl_vector * dx, const gsl_vector * x,
                                 double epsabs, double epsrel) nogil

    int gsl_multifit_test_gradient (const gsl_vector * g, double epsabs) nogil

    int gsl_multifit_fdfsolver_dif_df (const gsl_vector * x, gsl_multifit_function_fdf * fdf,
                                       const gsl_vector * f, gsl_matrix * J) nogil

    int gsl_multifit_fdfsolver_dif_fdf (const gsl_vector * x, gsl_multifit_function_fdf * fdf,
                                        gsl_vector * f, gsl_matrix * J) nogil

    const gsl_multifit_fdfsolver_type * gsl_multifit_fdfsolver_lmder
    const gsl_multifit_fdfsolver_type * gsl_multifit_fdfsolver_lmsder
