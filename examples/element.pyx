from cython_gsl cimport *

def main():
    x = 0.5; y = 0.9
    cdef gsl_sf_result res
    print gsl_sf_multiply_e(x,y, &res)
    print res.val, res.err
