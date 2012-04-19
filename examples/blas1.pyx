from cython_gsl cimport *

def main():
    cdef int i
    cdef double res
    cdef gsl_vector *v1, *v2
    v1 = gsl_vector_alloc(10)
    v2 = gsl_vector_alloc(10)

    for i from 0 <= i < 10:
        gsl_vector_set (v1, i, 1)

    for i from 0 <= i < 10:
        gsl_vector_set (v2, i, 2)

    gsl_blas_ddot(v1, v2, &res)
    print res
