from cython_gsl cimport gsl
cimport libc.math

ctypedef double * double_ptr
ctypedef void * void_ptr

cdef double normal(double x, void * params) nogil:
    cdef double mu = (<double_ptr> params)[0]
    cdef double sigma = (<double_ptr> params)[1]

    return gsl.gsl_ran_gaussian_pdf(x, sigma) + mu

cdef double cdf_numerical(double x, double mu, double sigma):
    cdef double alpha, result, error, expected
    cdef gsl.gsl_integration_workspace * W
    W = gsl.gsl_integration_workspace_alloc(1000)
    cdef gsl.gsl_function F
    cdef double params[1]
    cdef size_t neval

    params[0] = mu
    params[1] = sigma

    F.function = &normal
    F.params = params

    gsl.gsl_integration_qag(&F, -libc.math.INFINITY, x, 1e-2, 1e-2, 1000, GSL_INTEG_GAUSS15, W, &result, &error)
    gsl.gsl_integration_workspace_free(W)

    return result
