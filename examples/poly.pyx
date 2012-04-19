from cython_gsl cimport *

def main():
    cdef int i
    cdef double a[6]
    cdef double z[10]
    cdef gsl_poly_complex_workspace * w
    a[0] = -1
    a[1] = a[2] = a[3] = a[4] = 0
    a[5] = 1
    w = gsl_poly_complex_workspace_alloc (6)
    gsl_poly_complex_solve (a, 6, w, z)
    gsl_poly_complex_workspace_free (w)

    for i from 0 <= i < 5:
        print "z%d = %+.18f %+.18f\n" % (i, z[2*i], z[2*i+1])
