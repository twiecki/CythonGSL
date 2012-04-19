from cython_gsl cimport *

def main ():
    cdef int i, n
    n = 4
    cdef double x[4], y[4], w[4]
    x[0] =  1970
    x[1] = 1980
    x[2] = 1990
    x[3] = 2000
    y[0] = 12
    y[1] = 11
    y[2] = 14
    y[3] = 13
    w[0] = 0.1
    w[1] = 0.2
    w[2] = 0.3
    w[3] = 0.4

    cdef double c0, c1, cov00, cov01, cov11, chisq

    gsl_fit_wlinear (x, 1, w, 1, y, 1, n,
                     &c0, &c1, &cov00, &cov01, &cov11,
                     &chisq)

    print "# best fit: Y = %g + %g X\n" % (c0, c1)
    print "# covariance matrix:\n"
    print "# [ %g, %g\n#   %g, %g]\n" % (cov00, cov01, cov01, cov11)
    print "# chisq = %g\n", chisq

    for i from 0 <= i < n:
        print "data: %g %g %g\n" % ( x[i], y[i], 1/sqrt(w[i]))

    print "\n"

    cdef double xf, yf, yf_err
    for i from -30 <= i < 130:
        xf = x[0] + (i/100.0) * (x[n-1] - x[0])

        gsl_fit_linear_est (xf, c0, c1, cov00, cov01, cov11, &yf, &yf_err)

        print "fit: %g %g\n" %(xf, yf)
        print "hi : %g %g\n" %(xf, yf + yf_err)
        print "lo : %g %g\n" %(xf, yf - yf_err)
