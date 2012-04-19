from cython_gsl cimport *

def main():
    cdef gsl_rng_type * T
    cdef gsl_rng * r
    cdef int i, n
    cdef double mu
    n = 10
    mu = 3.0


    gsl_rng_env_setup()

    T = gsl_rng_default
    r = gsl_rng_alloc (T)

    cdef unsigned int k
    for i  from 0 <= i < n:
        k = gsl_ran_poisson (r, mu)
        print " %u" % k,

    print "\n"
