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
    cdef double mu
    mu = 10
    cdef gsl_odeiv2_system sys
    sys.function = func
    sys.jacobian = jac
    sys.dimension = 2
    sys.params = &mu

    cdef gsl_odeiv2_driver * d
    d = gsl_odeiv2_driver_alloc_y_new(
        &sys, gsl_odeiv2_step_rk8pd,
        1e-6, 1e-6, 0.0)

    cdef int i
    cdef double t, t1, y[2]
    t = 0.0
    t1 = 100.0
    y[0] = 1.0
    y[1] = 0.0

    cdef int status
    cdef double ti
    for i from 1 <= i <= 100:
        ti = i * t1 / 100.0
        status = gsl_odeiv2_driver_apply (d, &t, ti, y)

        if (status != GSL_SUCCESS):
            print("error, return value=%d\n" % status)
            break

        print("%.5e %.5e %.5e\n" %(t, y[0], y[1]))

    gsl_odeiv2_driver_free(d)

