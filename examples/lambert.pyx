from cython_gsl cimport *

def main():
    print gsl_sf_lambert_W0(0.5)
