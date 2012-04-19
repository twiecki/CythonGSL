from cython_gsl cimport *

def main():
    print gsl_sf_hyperg_0F1(0.5, 0.6)
