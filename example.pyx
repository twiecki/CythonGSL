from cython_gsl import gsl

ctypedef double * double_ptr
ctypedef void * void_ptr

cdef extern from "stdlib.h":
    void free(void* ptr)
    void* malloc(size_t size)
    void* realloc(void* ptr, size_t size)

cpdef Normal(double x, double sigma):
    return gsl.gsl_ran_gaussian_pdf(x, sigma)

