from cython_gsl cimport *

cdef extern from "stdlib.h":
    void *malloc(size_t size)
    int free(void*)
    int sizeof()

def main():
    cdef gsl_rng_type * T
    cdef gsl_rng * r

    cdef size_t i, k, N
    k = 5
    N= 100000
    cdef double * x, * small
    x = <double *> malloc (N * sizeof(double))
    small = <double *> malloc (k * sizeof(double))

    gsl_rng_env_setup()

    T = gsl_rng_default
    r = gsl_rng_alloc (T)

    for i from 0 <= i < N:
        x[i] = gsl_rng_uniform(r)

    gsl_sort_smallest (small, k, x, 1, N)

    print "%d smallest values from %d" %(k, N)

    for i from 0 <= i < k:
        print "%d: %.18f\n" %(i, small[i]),
