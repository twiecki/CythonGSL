from cython_gsl cimport *

cdef double fn1(double x, void * params) nogil:
    return cos(x) + 1.0

def main():
    cdef int status, iter, max_iter
    iter = 0
    max_iter = 100
    cdef gsl_min_fminimizer_type *T
    cdef gsl_min_fminimizer *s
    cdef double m, m_expected, a, b
    m = 2.0
    m_expected = M_PI
    a = 0.0
    b = 6.0
    cdef gsl_function F

    F.function = &fn1
    F.params = NULL

    T = gsl_min_fminimizer_brent
    s = gsl_min_fminimizer_alloc(T)
    gsl_min_fminimizer_set(s, &F, m, a, b)

    print "using %s method\n" % gsl_min_fminimizer_name(s)

    print "%5s [%9s, %9s] %9s %10s %9s\n" % \
          ("iter", "lower", "upper", "min", "err", "err(est)")

    print "%5d [%.7f, %.7f] %.7f %+.7f %.7f\n" % \
          (iter, a, b, m, m - m_expected, b - a)

    cdef double GSL_CONTINUE
    GSL_CONTINUE = -2  # to go...
    status = -2

    while (status == GSL_CONTINUE and iter < max_iter):
        iter = iter + 1
        status = gsl_min_fminimizer_iterate(s)

        m = gsl_min_fminimizer_x_minimum(s)
        a = gsl_min_fminimizer_x_lower(s)
        b = gsl_min_fminimizer_x_upper(s)

        status = gsl_min_test_interval(a, b, 0.001, 0.0)

        if (status == GSL_SUCCESS):
            print ("Converged:\n")

        print "%5d [%.7f, %.7f] %.7f %.7f %+.7f %.7f\n" % \
              (iter, a, b, m, m_expected, m - m_expected, b - a)

    return status

