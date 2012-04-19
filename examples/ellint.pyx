from cython_gsl cimport *

def main():
    x = 0.6
    r = gsl_sf_ellint_Kcomp(x, GSL_PREC_DOUBLE)
    print "%.18f\n" % r
