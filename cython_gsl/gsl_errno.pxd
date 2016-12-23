cdef extern from "gsl/gsl_errno.h":
    ctypedef void gsl_error_handler_t(const char * reason, const char * file,
                                      int line, int gsl_errno)
    gsl_error_handler_t * gsl_set_error_handler (gsl_error_handler_t * new_handler) nogil
    gsl_error_handler_t * gsl_set_error_handler_off () nogil
