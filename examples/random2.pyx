from cython_gsl cimport *

def main():
    cdef double P, Q
    cdef double x
    x = 2.0

    P = gsl_cdf_ugaussian_P (x)
    print "prob(x < %f) = %f\n"  % (x, P)

    Q = gsl_cdf_ugaussian_Q (x)
    print "prob(x < %f) = %f\n" % (x, Q)

    x = gsl_cdf_ugaussian_Pinv (P)
    print "Pinv(%f) = %f\n" % (P, x)

    x = gsl_cdf_ugaussian_Qinv (Q)
    print "Qinv(%f) = %f\n" % (Q, x)
