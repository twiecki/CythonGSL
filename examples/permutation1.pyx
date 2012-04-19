from cython_gsl cimport *

def main():

    N = 10
    cdef gsl_rng_type * T
    cdef gsl_rng * r

    cdef gsl_permutation * p, * q
    p = gsl_permutation_alloc (N)
    q = gsl_permutation_alloc (N)

    gsl_rng_env_setup()
    T = gsl_rng_default
    r = gsl_rng_alloc (T)

    print "initial permutation:"
    gsl_permutation_init (p)
    gsl_permutation_fprintf (stdout, p, " %u")
    print "\n"

    print " random permutation:"
    gsl_ran_shuffle (r, p.data, N, sizeof(size_t))
    gsl_permutation_fprintf (stdout, p, " %u")
    print "\n"

    print "inverse permutation:"
    gsl_permutation_inverse (q, p)
    gsl_permutation_fprintf (stdout, q, " %u")
    print "\n"
