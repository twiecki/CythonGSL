from cython_gsl cimport *

def main():
    cdef double x
    x = 1.7
    print gsl_sf_clausen(x)
