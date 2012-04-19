from cython_gsl cimport *

def main():
    cdef double a_data[16], b_data[4]
    cdef gsl_matrix_view m
    cdef gsl_vector_view b
    cdef gsl_vector *x
    cdef int s
    cdef gsl_permutation * p

    a_data[0] = 0.18
    a_data[1] = 0.60
    a_data[2] = 0.57
    a_data[3] = 0.96
    a_data[4] = 0.41
    a_data[5] = 0.24
    a_data[6] = 0.99
    a_data[7] = 0.58
    a_data[8] = 0.14
    a_data[9] = 0.30
    a_data[10] = 0.97
    a_data[11] = 0.66
    a_data[12] = 0.51
    a_data[13] = 0.13
    a_data[14] = 0.19
    a_data[15] = 0.85

    b_data[0] = 1
    b_data[1] = 2
    b_data[2] = 3
    b_data[3] = 4


    m = gsl_matrix_view_array (a_data, 4, 4)
    b = gsl_vector_view_array (b_data, 4)
    x = gsl_vector_alloc (4)
    p = gsl_permutation_alloc (4)

    gsl_vector_fprintf (stdout,&b.vector, "%g")

    gsl_linalg_LU_decomp (&m.matrix, p, &s)
    gsl_linalg_LU_solve (&m.matrix, p, &b.vector, x)

    print "x = \n"
    gsl_vector_fprintf (stdout, x, "%g")

    gsl_permutation_free (p)
