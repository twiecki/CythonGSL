#cython: cdivision=True

from cython_gsl cimport *

cdef double exact
exact = 1.3932039296856768591842462603255

cdef double g (double *k, size_t dim, void *params) nogil:
    cdef double A
    A = 1.0 / (M_PI * M_PI * M_PI)
    return A / (1.0 - cos (k[0]) * cos (k[1]) * cos (k[2]))


cdef void display_results (char *title, double result, double error):
    print "%s ==================\n" % title
    print "result = % .6f\n" % result
    print "sigma  = % .6f\n" % error
    print "exact  = % .6f\n" % exact
    print "error  = % .6f = %.1g sigma\n" % (result - exact,
           float(result - exact) / error)


def main ():
    cdef double res, err

    cdef double xl[3]
    xl[0] = 0
    xl[1] = 0
    xl[2] = 0
    cdef double xu[3]
    xu[0] = M_PI
    xu[1] = M_PI
    xu[2] = M_PI

    cdef gsl_rng_type *T
    cdef gsl_rng *r

    cdef gsl_monte_function G
    G.f =  &g
    G.dim = 3
    G.params = NULL

    cdef size_t calls
    calls = 500000

    gsl_rng_env_setup ()

    T = gsl_rng_default
    r = gsl_rng_alloc (T)

    cdef gsl_monte_plain_state *s
    s = gsl_monte_plain_alloc (3)
    gsl_monte_plain_integrate (&G, xl, xu, 3, calls, r, s,
                                 &res, &err)
    gsl_monte_plain_free (s)

    display_results ("plain", res, err)




    cdef gsl_monte_miser_state *sm
    sm = gsl_monte_miser_alloc (3)
    gsl_monte_miser_integrate (&G, xl, xu, 3, calls, r, sm,
                                 &res, &err)
    gsl_monte_miser_free (sm)

    display_results ("miser", res, err)

    print "DB: start vegas"

    cdef gsl_monte_vegas_state *sv
    sv = gsl_monte_vegas_alloc (3)

    gsl_monte_vegas_integrate (&G, xl, xu, 3, 10000, r, sv,
                                 &res, &err)
    display_results ("vegas warm-up", res, err)

    print "converging...\n"

    while (float(sv.chisq - 1.0) > 0.5):
        gsl_monte_vegas_integrate (&G, xl, xu, 3, calls/5, r, sv,
                                       &res, &err)
        print "result = % .6f sigma = % .6f chisq/dof = %.1f\n" % \
              (res, err, sv.chisq)

    display_results ("vegas final", res, err)

    gsl_monte_vegas_free (sv)
