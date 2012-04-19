from cython_gsl cimport *

cdef double f (double x, void *p) nogil:
    if (x < 0.5):
        return 0.25
    else:
        return 0.75


def main ():
    cdef int i, n
    n = 10000

    cdef gsl_cheb_series *cs
    cs = gsl_cheb_alloc (40)

    cdef gsl_function F

    F.function = f
    F.params = NULL

    gsl_cheb_init (cs, &F, 0.0, 1.0)

    cdef double x, r10, r40
    for i from 0 <= i < n:
        x = i / <double>n
        r10 = gsl_cheb_eval_n (cs, 10, x)
        r40 = gsl_cheb_eval (cs, x)
        print "%g %g %g %g\n" % \
                (x, GSL_FN_EVAL (&F, x), r10, r40)

    gsl_cheb_free (cs)
