from cython_gsl cimport *

def main():
    cdef double x
    x = 5.0
    cdef gsl_sf_result result
    cdef int status
    status = gsl_sf_bessel_J0_e (x, &result)
    print "J0(5.0) = %.18f\n      +/- % .18f\n" % (result.val, result.err)
    print "%.18f\n" % gsl_sf_bessel_J0 (5.0)
