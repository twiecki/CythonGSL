from cython_gsl cimport *


def main():
    cdef int i
    cdef double data[2*128]

    for i from 0 <= i < 128:
        data[2*i] = 0.0
        data[2*i+1] = 0.0

    data[0] = 1.0

    for i from 1 <= i <= 10:
        data[2*i] = data[2*(128-i)] = 1.0

    for i from 0 <= i < 128:
        print "%d %e %e\n" %(i, data[2*i], data[2*i+1])

    gsl_fft_complex_radix2_forward (data, 1, 128)

    for i from 0 <= i < 128:
        print "%d %e %e\n" %(i, data[2*i]/sqrt(128), data[2*i+1]/sqrt(128))
