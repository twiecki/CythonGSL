from cython_gsl cimport *

def main():
    print gsl_sf_zeta_int(3)
