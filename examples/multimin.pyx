from cython_gsl cimport *

cdef double my_f (const gsl_vector * v, void * params) nogil:
    cdef double x, y
    cdef double * p = <double *> params

    x = gsl_vector_get(v, 0)
    y = gsl_vector_get(v, 1)

    return p[2] * (x - p[0]) * (x - p[0]) + p[3] * (y - p[1]) * (y - p[1]) + p[4]

cdef void my_df (const gsl_vector * v, void * params, gsl_vector * df) nogil:
    cdef double x, y
    cdef double * p = <double *> params

    x = gsl_vector_get(v, 0)
    y = gsl_vector_get(v, 1)

    gsl_vector_set(df, 0, 2.0 * p[2] * (x - p[0]))
    gsl_vector_set(df, 1, 2.0 * p[3] * (y - p[1]))

cdef void my_fdf (const gsl_vector * x, void * params, double * f, gsl_vector * df) nogil:
    f[0] = my_f (x, params)
    my_df (x, params, df)

def main_fdf():
    cdef size_t iter = 0
    cdef int max_iter = 100
    cdef int status

    cdef const gsl_multimin_fdfminimizer_type * T
    cdef gsl_multimin_fdfminimizer * s
    cdef gsl_vector * x

    cdef gsl_multimin_function_fdf my_func
    cdef double p[5]
    p[0] = 1.0
    p[1] = 2.0
    p[2] = 10.0
    p[3] = 20.0
    p[4] = 30.0
    my_func.n = 2
    my_func.f = &my_f
    my_func.df = &my_df
    my_func.fdf = &my_fdf
    my_func.params = <void *> p

    # Starting point, x = (5, 7)
    x = gsl_vector_alloc (2)
    gsl_vector_set (x, 0, 5.0)
    gsl_vector_set (x, 1, 7.0)

    T = gsl_multimin_fdfminimizer_conjugate_fr
    s = gsl_multimin_fdfminimizer_alloc (T, 2)

    gsl_multimin_fdfminimizer_set (s, &my_func, x, 0.01, 1e-4)

    status = GSL_CONTINUE

    while (status == GSL_CONTINUE and iter <= max_iter):
        iter += 1
        status = gsl_multimin_fdfminimizer_iterate (s)

        if status:
            break

        status = gsl_multimin_test_gradient (s.gradient, 1e-3)

        if status == GSL_SUCCESS:
            print("Minimum found at:\n")

        print("%5d %.5f %.5f %10.5f\n" %\
              (iter, gsl_vector_get (s.x, 0), gsl_vector_get (s.x, 1), s.f))

    gsl_multimin_fdfminimizer_free (s)
    gsl_vector_free (x)

    return status

def main_f():
    cdef size_t iter = 0
    cdef int max_iter = 100
    cdef int status

    cdef const gsl_multimin_fminimizer_type * T
    cdef gsl_multimin_fminimizer * s
    cdef gsl_vector * ss
    cdef gsl_vector * x

    cdef gsl_multimin_function my_func
    cdef double p[5]
    p[0] = 1.0
    p[1] = 2.0
    p[2] = 10.0
    p[3] = 20.0
    p[4] = 30.0
    my_func.n = 2
    my_func.f = &my_f
    my_func.params = <void *> p

    # Starting point, x = (5, 7)
    x = gsl_vector_alloc (2)
    gsl_vector_set (x, 0, 5.0)
    gsl_vector_set (x, 1, 7.0)

    # Set initial step sizes to 1
    ss = gsl_vector_alloc (2)
    gsl_vector_set_all (ss, 1.0)

    T = gsl_multimin_fminimizer_nmsimplex2
    s = gsl_multimin_fminimizer_alloc (T, 2)

    gsl_multimin_fminimizer_set (s, &my_func, x, ss)

    status = GSL_CONTINUE

    while (status == GSL_CONTINUE and iter <= max_iter):
        iter += 1
        status = gsl_multimin_fminimizer_iterate (s)

        if status:
            print(status)
            break

        size = gsl_multimin_fminimizer_size (s)
        status = gsl_multimin_test_size (size, 1e-2)

        if status == GSL_SUCCESS:
            print("Minimum found at:\n")

        print("%5d %10.3e %10.3e f() = %7.3f size = %.3f\n" %\
              (iter, gsl_vector_get (s.x, 0), gsl_vector_get (s.x, 1), s.fval, size))

    gsl_multimin_fminimizer_free (s)
    gsl_vector_free (x)
    gsl_vector_free (ss)

    return status
