from cython_gsl cimport *


def main():
    cdef int i, n
    n = 630
    cdef double data[2*630]

    cdef gsl_fft_complex_wavetable * wavetable
    cdef gsl_fft_complex_workspace * workspace

    for i from 0 <= i < n:
        data[2*i] = 0.0
        data[2*i+1] = 0.0

    data[0] = 1.0
    for i from 1 <= i <= 10:
        data[2*i] = data[2*(n-i)] = 1.0

    for i from 0 <= i < n:
        print "%d: %e %e\n" %(i, data[2*i], data[2*i+1]),
    print "\n"

    wavetable = gsl_fft_complex_wavetable_alloc (n)
    workspace = gsl_fft_complex_workspace_alloc (n)

    for i from 0 <= i < wavetable.nf:
        print "# factor %d: %d\n" %(i, wavetable.factor[i]),
    print "\n"

    gsl_fft_complex_forward (data, 1, n, wavetable, workspace)

    for i from 0 <= i < n:
        print "%d: %e %e\n" %(i, data[2*i], data[2*i+1]),
    print "\n"
    gsl_fft_complex_wavetable_free (wavetable)
    gsl_fft_complex_workspace_free (workspace)
