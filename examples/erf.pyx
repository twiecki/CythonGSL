from cython_gsl cimport *

def main():
    print gsl_sf_erf(0.5)
