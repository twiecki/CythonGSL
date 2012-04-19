from cython_gsl cimport *

def main():
    cdef double data[16]
    cdef gsl_matrix_view m
    cdef gsl_vector *eval
    cdef gsl_matrix *evec
    cdef gsl_eigen_symmv_workspace *w

    data[0] = 1.0
    data[1] = 1/2.0
    data[2] = 1/3.0
    data[3] =  1/4.0

    data[4] = 1/2.0
    data[5] = 1/3.0
    data[6] = 1/4.0
    data[7] = 1/5.0

    data[8] = 1/3.0
    data[9] = 1/4.0
    data[10] = 1/5.0
    data[11] = 1/6.0

    data[12] = 1/4.0
    data[13] = 1/5.0
    data[14] = 1/6.0
    data[15] = 1/7.0

    m = gsl_matrix_view_array (data, 4, 4)
    eval = gsl_vector_alloc (4)
    evec = gsl_matrix_alloc (4, 4)
    w = gsl_eigen_symmv_alloc (4)
    gsl_eigen_symmv (&(m.matrix), eval, evec, w)
    gsl_eigen_symmv_free (w)
    gsl_eigen_symmv_sort (eval, evec, GSL_EIGEN_SORT_ABS_ASC)

    cdef int i
    cdef double eval_i
    cdef gsl_vector_view evec_i
    for i from 0 <= i < 4:
        eval_i = gsl_vector_get (eval, i)
        evec_i = gsl_matrix_column (evec, i)
        print "eigenvalue = %g\n" % eval_i
        print "eigenvector = \n"
        gsl_vector_fprintf (stdout, &evec_i.vector, "%g")
