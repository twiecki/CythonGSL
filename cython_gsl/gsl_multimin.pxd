from cython_gsl cimport *

cdef extern from "gsl/gsl_multimin.h":

    # Definition of an arbitrary real-valued function with gsl_vector input and
    # parameters
    ctypedef struct gsl_multimin_function:
        double (* f) (const gsl_vector * x, void * params) nogil
        size_t n
        void * params

    # Definition of an arbitrary differentiable real-valued function with
    # gsl_vector input and parameters
    ctypedef struct gsl_multimin_function_fdf:
        double (* f) (const gsl_vector * x, void * params) nogil
        void (* df) (const gsl_vector * x, void * params, gsl_vector * df) nogil
        void (* fdf) (const gsl_vector * x, void * params, double * f, gsl_vector * df) nogil
        size_t n
        void * params

    # Minimization of non-differentiable functions
    ctypedef struct gsl_multimin_fminimizer_type:
        const char * name
        int (* alloc) (void * state, size_t n) nogil
        int (* set) (void * state,
                     gsl_multimin_function * f,
                     const gsl_vector * x,
                     double * size,
                     const gsl_vector * step_size) nogil
        int (* iterate) (void * state,
                         gsl_multimin_function * f,
                         gsl_vector * x,
                         double * size,
                         double * fval) nogil
        void (* free) (void * state) nogil

    ctypedef struct gsl_multimin_fminimizer:
        const gsl_multimin_fminimizer_type * type
        gsl_multimin_function * f
        double fval
        gsl_vector * x
        double size
        void * state

    gsl_multimin_fminimizer * gsl_multimin_fminimizer_alloc (const gsl_multimin_fminimizer_type * T,
                                                             size_t n) nogil

    int gsl_multimin_fminimizer_set (gsl_multimin_fminimizer * s,
                                     gsl_multimin_function * f,
                                     const gsl_vector * x,
                                     const gsl_vector * step_size) nogil

    void gsl_multimin_fminimizer_free (gsl_multimin_fminimizer * s) nogil

    const char * gsl_multimin_fminimizer_name (const gsl_multimin_fminimizer * s) nogil

    int gsl_multimin_fminimizer_iterate (gsl_multimin_fminimizer * s) nogil

    gsl_vector * gsl_multimin_fminimizer_x (const gsl_multimin_fminimizer * s) nogil

    double gsl_multimin_fminimizer_minimum (const gsl_multimin_fminimizer * s) nogil

    double gsl_multimin_fminimizer_size (const gsl_multimin_fminimizer * s) nogil

    int gsl_multimin_test_gradient (const gsl_vector * g, double epsabs) nogil

    int gsl_multimin_test_size (const double size, double epsabs) nogil

    # Minimization of differentiable functions
    ctypedef struct gsl_multimin_fdfminimizer_type:
        const char * name
        size_t size
        int (* alloc) (void * state, size_t n) nogil
        int (* set) (void * state,
                     gsl_multimin_function_fdf * fdf,
                     const gsl_vector * x,
                     double * f,
                     gsl_vector * gradient,
                     double step_size,
                     double tol) nogil
        int (* iterate) (void * state,
                         gsl_multimin_function_fdf * fdf,
                         gsl_vector * x,
                         double * f,
                         gsl_vector * gradient,
                         gsl_vector * dx) nogil
        int (* restart) (void * state) nogil
        void (* free) (void * state) nogil

    ctypedef struct gsl_multimin_fdfminimizer:
        const gsl_multimin_fdfminimizer_type * type
        gsl_multimin_function_fdf * fdf

        double f
        gsl_vector * x
        gsl_vector * gradient
        gsl_vector * dx

        void * state

    gsl_multimin_fdfminimizer * gsl_multimin_fdfminimizer_alloc (const gsl_multimin_fdfminimizer_type * T,
                                                                 size_t n) nogil

    int gsl_multimin_fdfminimizer_set (gsl_multimin_fdfminimizer * s,
                                       gsl_multimin_function_fdf * fdf,
                                       const gsl_vector * x,
                                       double step_size,
                                       double tol) nogil

    void gsl_multimin_fdfminimizer_free (gsl_multimin_fdfminimizer *s) nogil

    const char * gsl_multimin_fdfminimizer_name (const gsl_multimin_fdfminimizer * s) nogil

    int gsl_multimin_fdfminimizer_iterate (gsl_multimin_fdfminimizer * s) nogil

    int gsl_multimin_fdfminimizer_restart (gsl_multimin_fdfminimizer * s) nogil

    gsl_vector * gsl_multimin_fdfminimizer_x (const gsl_multimin_fdfminimizer * s) nogil

    gsl_vector * gsl_multimin_fdfminimizer_dx (const gsl_multimin_fdfminimizer * s) nogil

    gsl_vector * gsl_multimin_fdfminimizer_gradient (const gsl_multimin_fdfminimizer * s) nogil

    double gsl_multimin_fdfminimizer_minimum (const gsl_multimin_fdfminimizer * s) nogil

    const gsl_multimin_fdfminimizer_type * gsl_multimin_fdfminimizer_steepest_descent
    const gsl_multimin_fdfminimizer_type * gsl_multimin_fdfminimizer_conjugate_pr
    const gsl_multimin_fdfminimizer_type * gsl_multimin_fdfminimizer_conjugate_fr
    const gsl_multimin_fdfminimizer_type * gsl_multimin_fdfminimizer_vector_bfgs
    const gsl_multimin_fdfminimizer_type * gsl_multimin_fdfminimizer_vector_bfgs2
    const gsl_multimin_fminimizer_type * gsl_multimin_fminimizer_nmsimplex
    const gsl_multimin_fminimizer_type * gsl_multimin_fminimizer_nmsimplex2
    const gsl_multimin_fminimizer_type * gsl_multimin_fminimizer_nmsimplex2rand
