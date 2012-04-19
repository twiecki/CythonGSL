from cython_gsl cimport *

def main():
    print gsl_sf_legendre_P1(0.5)
