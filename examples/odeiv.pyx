from cython_gsl cimport *


cdef int func (double t,  double y[], double f[], void *params) nogil:
    cdef double mu
    mu = (<double *> params)[0]
    f[0] = y[1]
    f[1] = -y[0] - mu*y[1]*(y[0]*y[0] - 1)
    return GSL_SUCCESS

cdef int jac (double t,  double y[], double *dfdy, double dfdt[], void *params) nogil:
    cdef double mu
    mu = (<double *>params)[0]
    cdef gsl_matrix_view dfdy_mat
    dfdy_mat  = gsl_matrix_view_array (dfdy, 2, 2)
    cdef gsl_matrix * m
    m = &dfdy_mat.matrix
    gsl_matrix_set (m, 0, 0, 0.0)
    gsl_matrix_set (m, 0, 1, 1.0)
    gsl_matrix_set (m, 1, 0, -2.0*mu*y[0]*y[1] - 1.0)
    gsl_matrix_set (m, 1, 1, -mu*(y[0]*y[0] - 1.0))
    dfdt[0] = 0.0
    dfdt[1] = 0.0
    return GSL_SUCCESS

def main ( ):
    cdef gsl_odeiv_step_type * T
    T = gsl_odeiv_step_rk8pd

    cdef gsl_odeiv_step * s
    s  = gsl_odeiv_step_alloc (T, 2)
    cdef gsl_odeiv_control * c
    c  = gsl_odeiv_control_y_new (1e-6, 0.0)
    cdef gsl_odeiv_evolve * e
    e  = gsl_odeiv_evolve_alloc (2)

    cdef double mu
    mu = 10
    cdef gsl_odeiv_system sys
    sys.function = func
    sys.jacobian = jac
    sys.dimension = 2
    sys.params = &mu

    cdef double t, t1, h, y[2]
    t = 0.0
    t1 = 100.0
    h = 1e-6
    y[0] = 1.0
    y[1] = 0.0

    cdef int status
    while (t < t1):
        status = gsl_odeiv_evolve_apply (e, c, s, &sys, &t, t1, &h, y)

        if (status != GSL_SUCCESS):
            break

        print "%.5e %.5e %.5e\n" %(t, y[0], y[1])

    gsl_odeiv_evolve_free (e)
    gsl_odeiv_control_free (c)
    gsl_odeiv_step_free (s)
