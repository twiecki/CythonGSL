cimport cython_gsl

ctypedef double * double_ptr
ctypedef void * void_ptr

cdef extern from "stdlib.h":
    void free(void* ptr)
    void* malloc(size_t size)
    void* realloc(void* ptr, size_t size)

cpdef Normal(double x, double sigma):
    return cython_gsl.gsl_ran_gaussian_pdf(x, sigma)

