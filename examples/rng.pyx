from cython_gsl cimport *

def main():
    cdef gsl_rng_type * T
    cdef gsl_rng * r

    cdef int i, n
    n = 10

    gsl_rng_env_setup()
    T = gsl_rng_default
    r = gsl_rng_alloc (T)

    cdef double u
    for i from 0 <= i < n:
        u = gsl_rng_uniform (r)
        print "%.5f\n" % u

    gsl_rng_free (r)
