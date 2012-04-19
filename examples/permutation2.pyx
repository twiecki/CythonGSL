from cython_gsl cimport *

def main():
    cdef gsl_permutation * p
    p = gsl_permutation_alloc (3)
    gsl_permutation_init (p)
    gsl_permutation_fprintf (stdout, p, " %u")
    print
    # GSL_SUCCESS = 0
    while (gsl_permutation_next(p) == GSL_SUCCESS):
        gsl_permutation_fprintf (stdout, p, " %u")
        print
    gsl_permutation_fprintf (stdout, p, " %u")
    print
