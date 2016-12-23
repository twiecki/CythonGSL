from cython_gsl cimport *
from cython_gsl.gsl_errno cimport *


def t_gsl_matrix_alloc_fail():
    cdef gsl_matrix * m
    h = gsl_set_error_handler_off()
    #This will fail b/c cannot alloc dimension of 0
    m = gsl_matrix_alloc (0, 3)
    gsl_set_error_handler(h)

