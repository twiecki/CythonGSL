from cython_gsl cimport *

def main():
    cdef gsl_combination * c
    cdef size_t i
    print "All subsets of {0,1,2,3} by size:\n"
    for i from 0 <= i <= 4:
        c = gsl_combination_calloc (4, i)
        print "{",
        gsl_combination_fprintf (stdout, c, " %u")
        print "}"
        while(gsl_combination_next (c) == 0): # GSL_SUCCESS = 0
            print "{",
            gsl_combination_fprintf (stdout, c, " %u")
            print "}"
        print "{",
        gsl_combination_fprintf (stdout, c, " %u")
        print "}"
        gsl_combination_free (c)
