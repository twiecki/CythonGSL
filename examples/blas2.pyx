from cython_gsl cimport *

def main():
    cdef double a[6]

    a[0] = 0.11
    a[1] = 0.12
    a[2] = 0.13
    a[3] = 0.21
    a[4] = 0.22
    a[5] = 0.23

    cdef double b[6]
    b[0] = 1011
    b[1] = 1012
    b[2] = 1021
    b[3] = 1022
    b[4] = 1031
    b[5] = 1032

    cdef double c[4]
    c[0] = 0
    c[1] = 0
    c[3] = 0
    c[4] = 0

    cdef gsl_matrix_view A, B, C
    A = gsl_matrix_view_array(a, 2, 3)
    B = gsl_matrix_view_array(b, 3, 2)
    C = gsl_matrix_view_array(c, 2, 2)

    gsl_blas_dgemm (CblasNoTrans, CblasNoTrans,
                      1.0, &A.matrix, &B.matrix, 0.0, &C.matrix)

    print "[ %g, %g\n" %(c[0], c[1])
    print "  %g, %g ]\n" %(c[2], c[3])
