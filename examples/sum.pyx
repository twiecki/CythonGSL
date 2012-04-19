from cython_gsl cimport *

cdef enum:
    N = 20

def main ():
    cdef double t[N], sum_accel, err, sum
    sum = 0
    cdef int n

    cdef gsl_sum_levin_u_workspace * w
    w  = gsl_sum_levin_u_alloc (N)

    cdef double zeta_2
    zeta_2 = M_PI * M_PI / 6.0

    cdef double np1
    for n from 0 <= n < N:
        np1 = n + 1.0
        t[n] = 1.0 / (np1 * np1)
        sum = sum + t[n]

    gsl_sum_levin_u_accel (t, N, w, &sum_accel, &err)

    print "term-by-term sum = % .16f using %d terms\n" %(sum, N)

    print "term-by-term sum = % .16f using %d terms\n" % \
            (w.sum_plain, w.terms_used)

    print "exact value      = % .16f\n" % zeta_2
    print "accelerated sum  = % .16f using %d terms\n" % \
            (sum_accel, w.terms_used)

    print "estimated error  = % .16f\n" % err
    cdef double boh
    boh = sum_accel - zeta_2
    print "actual error     = % .16f\n" % boh
    #print "actual error     = % .16f\n" % sum_accel - zeta_2

    gsl_sum_levin_u_free (w)
