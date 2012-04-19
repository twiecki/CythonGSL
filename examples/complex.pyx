from cython_gsl cimport *

def main():
    cdef gsl_complex c2
    GSL_SET_COMPLEX(&c2, 2.2, 1.5)
    print 'c2', GSL_REAL(c2)
