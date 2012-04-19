from cython_gsl cimport *

def main():
    print "%.15f\n" % gsl_sf_airy_Ai(0, GSL_PREC_DOUBLE)
