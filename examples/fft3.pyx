from cython_gsl cimport *

def main():
    cdef int i, n
    n = 100
    cdef double data[100]

    cdef gsl_fft_real_wavetable * real
    cdef gsl_fft_halfcomplex_wavetable * hc
    cdef gsl_fft_real_workspace * work

    for i from 0 <= i < n:
        data[i] = 0.0

    for i from n/3 <= i < 2*n/3:
        data[i] = 1.0

    for i from 0 <= i < n:
        print "%d: %e\n" %(i, data[i]),
    print "\n"

    work = gsl_fft_real_workspace_alloc (n)
    real = gsl_fft_real_wavetable_alloc (n)

    gsl_fft_real_transform (data, 1, n, real, work)

    gsl_fft_real_wavetable_free (real)

    for i from 11 <= i < n:
        data[i] = 0

    hc = gsl_fft_halfcomplex_wavetable_alloc (n)

    gsl_fft_halfcomplex_inverse (data, 1, n, hc, work)
    gsl_fft_halfcomplex_wavetable_free (hc)

    for i from 0 <= i < n:
        print "%d: %e\n" %(i, data[i]),

    gsl_fft_real_workspace_free (work)
