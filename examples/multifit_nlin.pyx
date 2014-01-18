from cython_gsl cimport *

from libc.math cimport exp, sqrt, pow
from libc.stdio cimport printf

ctypedef struct Data:
    size_t n
    double * y
    double * sigma

cdef int expb_f (const gsl_vector * x, void * data, gsl_vector * f) nogil:
    cdef Data * d = <Data *> data
    cdef size_t n = d.n
    cdef double * y = d.y
    cdef double * sigma = d.sigma

    cdef double A = gsl_vector_get (x, 0)
    cdef double lambd = gsl_vector_get (x, 1)
    cdef double b = gsl_vector_get (x, 2)

    cdef size_t i
    cdef double t, Yi

    for i from 0 <= i < n:
        # Model Yi = A * exp(-lambd * i) + b
        t = i
        Yi = A * exp (-lambd * t) + b
        gsl_vector_set (f, i, (Yi - y[i])/sigma[i])

    return GSL_SUCCESS

cdef int expb_df (const gsl_vector * x, void * data, gsl_matrix * J) nogil:
    cdef Data * d = <Data *> data
    cdef size_t n = d.n
    cdef double * y = d.y
    cdef double * sigma = d.sigma

    cdef double A = gsl_vector_get (x, 0)
    cdef double lambd = gsl_vector_get (x, 1)

    cdef size_t i
    cdef double t, s, e

    for i from 0 <= i < n:
        # Jacobian matrix J(i,j) = dfi / dxj,
        # where fi = (Yi - yi)/sigma[i],
        #       Yi = A * exp(-lambda * i) + b
        # and the xj are the parameters (A, lambda, b)
        t = i
        s = sigma[i]
        e = exp (-lambd * t)
        gsl_matrix_set (J, i, 0, e/s)
        gsl_matrix_set (J, i, 1, -t * A * e/s)
        gsl_matrix_set (J, i, 2, 1/s)

    return GSL_SUCCESS

cdef int expb_fdf (const gsl_vector * x, void * data, gsl_vector * f,
                   gsl_matrix * J) nogil:

    expb_f (x, data, f)
    expb_df (x, data, J)

    return GSL_SUCCESS

cdef void print_state (size_t iter, gsl_multifit_fdfsolver * s) nogil:
    printf ("iter: %3u x = % 15.8f % 15.8f % 15.8f |f(x)| = %g\n",
            iter,
            gsl_vector_get (s.x, 0), 
            gsl_vector_get (s.x, 1),
            gsl_vector_get (s.x, 2), 
            gsl_blas_dnrm2 (s.f))

def main():
    cdef const gsl_multifit_fdfsolver_type * T
    cdef gsl_multifit_fdfsolver * s
    cdef int status = GSL_CONTINUE
    cdef unsigned int i
    cdef unsigned int iter = 0
    cdef size_t n = 40
    cdef size_t p = 3

    cdef gsl_matrix * covar = gsl_matrix_alloc (p, p)
    cdef double y[40]
    cdef double sigma[40]
    cdef Data d
    d.n = n
    d.y = y
    d.sigma = sigma

    cdef gsl_multifit_function_fdf f
    cdef double * x_init = [1.0, 0.0, 0.0]
    cdef gsl_vector_view x = gsl_vector_view_array (x_init, p)
    cdef const gsl_rng_type * type
    cdef gsl_rng * r

    gsl_rng_env_setup ()

    type = gsl_rng_default
    r = gsl_rng_alloc (type)

    f.f = &expb_f
    f.df = &expb_df
    f.fdf = &expb_fdf
    f.n = n
    f.p = p
    f.params = &d

    # This is the data to be fitted
    cdef double t

    for i from 0 <= i < n:
        t = i;
        y[i] = 1.0 + 5 * exp (-0.1 * t) + gsl_ran_gaussian (r, 0.1)
        sigma[i] = 0.1
        printf ("data: %u %g %g\n", i, y[i], sigma[i])

    T = gsl_multifit_fdfsolver_lmsder
    s = gsl_multifit_fdfsolver_alloc (T, n, p)
    gsl_multifit_fdfsolver_set (s, &f, &x.vector)

    print_state (iter, s)

    while (status == GSL_CONTINUE and iter < 500):

        iter += 1
        status = gsl_multifit_fdfsolver_iterate (s)

        printf ("status = %d\n", status)

        print_state (iter, s)

        if status:
            break

        status = gsl_multifit_test_delta (s.dx, s.x, 1e-4, 1e-4)

    gsl_multifit_covar (s.J, 0.0, covar)

    cdef double chi = gsl_blas_dnrm2 (s.f)
    cdef double dof = n - p
    cdef double c = GSL_MAX_DBL(1, chi / sqrt (dof))

    printf("chisq/dof = %g\n",  pow (chi, 2.0) / dof)

    cdef double fit, err

    fit = gsl_vector_get (s.x, 0)
    err = sqrt (gsl_matrix_get (covar, 0, 0))
    printf ("A      = %.5f +/- %.5f\n", fit, c*err)

    fit = gsl_vector_get (s.x, 1)
    err = sqrt (gsl_matrix_get (covar, 1, 1))
    printf ("lambda = %.5f +/- %.5f\n", fit, c*err)

    fit = gsl_vector_get (s.x, 2)
    err = sqrt (gsl_matrix_get (covar, 2, 2))
    printf ("b      = %.5f +/- %.5f\n", fit, c*err)

    printf ("status = %d\n", status)

    gsl_multifit_fdfsolver_free (s)
    gsl_matrix_free (covar)
    gsl_rng_free (r)
