#cython: cdivision=True

from cython_gsl cimport *

ctypedef double * double_ptr
ctypedef void * void_ptr

cdef double foo(double x, void * params) nogil:
    cdef double alpha, f
    alpha = (<double_ptr> params)[0]
    f = log(alpha*x) / sqrt(x)
    return f


def main():
    cdef gsl_integration_workspace * w
    cdef double result, error, expected, alpha
    w = gsl_integration_workspace_alloc (1000)

    expected = -4.0
    alpha = 1

    cdef gsl_function F
    F.function = &foo
    F.params = &alpha

    gsl_integration_qags (&F, 0, 1, 0, 1e-7, 1000, w, &result, &error)
    print "result          = % .18f\n" % result
    print "estimated error          = % .18f\n" % error
