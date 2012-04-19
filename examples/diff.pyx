from cython_gsl cimport *

cdef double f (double x, void * params) nogil:
    return pow (x, 1.5)

def main ():
    cdef gsl_function F
    cdef double result, abserr

    F.function = &f
    F.params = NULL

    print "f(x) = x^(3/2)\n"

    gsl_diff_central (&F, 2.0, &result, &abserr)
    print "x = 2.0\n"
    print "f'(x) = %.10f +/- %.5f\n" % (result, abserr)
    print "exact = %.10f\n\n" % (1.5 * sqrt(2.0))

    gsl_diff_forward (&F, 0.0, &result, &abserr)
    print "x = 0.0\n"
    print "f'(x) = %.10f +/- %.5f\n" % (result, abserr)
    print "exact = %.10f\n" % 0.0
