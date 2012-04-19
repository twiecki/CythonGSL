from cython_gsl cimport *

def main():
    cdef int i
    cdef gsl_qrng * q
    q = gsl_qrng_alloc (gsl_qrng_sobol, 2)
    cdef double v[2]
    for i  from 0 <= i < 1024:
        gsl_qrng_get (q, v)
        print "%.5f %.5f\n" % (v[0], v[1])

    gsl_qrng_free (q)
