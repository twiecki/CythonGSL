from cython_gsl cimport *

def main():
    print gsl_sf_expint_E1(0.5)
