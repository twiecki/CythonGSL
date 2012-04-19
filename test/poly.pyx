from cython_gsl cimport *


def t_gsl_poly_eval():
    cdef double c[2]
    c[0] = 1; c[1] = 2
    # c[0] + c[1] x   for x = 3
    return gsl_poly_eval(c, 2, 3) - 7

def t_gsl_poly_dd_init():
    cdef double xa[4], ya[4], dd[4]
    xa[0]=0; xa[1]=2; xa[2]=4; xa[3]=6
    ya[0]=0; ya[1]=10; ya[2]=0; ya[3]=-10
    cdef double x01, x12, x23, x012
    x01 = (ya[0] - ya[1])/(xa[0] - xa[1])
    x12 = (ya[1] - ya[2])/(xa[1] - xa[2])
    x23 = (ya[2] - ya[3])/(xa[2] - xa[3])
    x012 = (x01 - x12)/(xa[0] - xa[2])
    # dd[3] not checked
    gsl_poly_dd_init(dd, xa, ya, 4)
    return (dd[0] -ya[0], dd[1] -x01, dd[2] - x012)


def t_gsl_poly_solve_quadratic():
    cdef double x0, x1
    gsl_poly_solve_quadratic(1, -3, 2, &x0, &x1)
    return (x0 - 1, x1 - 2)

def t_gsl_poly_complex_solve_quadratic():
    cdef gsl_complex z0, z1
    gsl_poly_complex_solve_quadratic(1, 0, 1, &z0, &z1)
    return(GSL_REAL(z0), GSL_IMAG(z0) +1, GSL_REAL(z1), GSL_IMAG(z1) -1)


def t_gsl_poly_solve_cubic1():
    cdef int roots
    cdef double x0, x1, x2
    roots = gsl_poly_solve_cubic(-5, 8, -4, &x0, &x1, &x2)
    return (roots - 3, x0 - 1, x1 - 2, x2 - 2)

def t_gsl_poly_solve_cubic2():
    cdef int roots
    cdef double x0, x1, x2
    roots = gsl_poly_solve_cubic(-2, 1, -2, &x0, &x1, &x2)
    return (roots - 1, x0 - 2)


def t_gsl_poly_complex_solve_cubic1():
    cdef gsl_complex z0, z1, z2
    gsl_poly_complex_solve_cubic(-5, 8, -4, &z0, &z1, &z2)
    return(GSL_REAL(z0) - 1, GSL_IMAG(z0), GSL_REAL(z1) -2 , GSL_IMAG(z1),
           GSL_REAL(z2) - 2, GSL_IMAG(z2))


def t_gsl_poly_complex_solve_cubic2():
    cdef gsl_complex z0, z1, z2
    gsl_poly_complex_solve_cubic(-2, 1, -2, &z0, &z1, &z2)
    return(GSL_REAL(z0), GSL_IMAG(z0) + 1, GSL_REAL(z1), GSL_IMAG(z1) - 1,
           GSL_REAL(z2) - 2, GSL_IMAG(z2) )

def t_gsl_poly_complex_solve():
    cdef double a[5], z[10]
    a[0] = 6; a[1] = -5; a[2] = 7; a[3] = -5; a[4] = 1
    cdef gsl_poly_complex_workspace * w
    w = gsl_poly_complex_workspace_alloc(5)
    gsl_poly_complex_solve(a, 5, w, z)
    gsl_poly_complex_workspace_free (w)
    return (z[0], z[1] - 1, z[2], z[3] + 1, z[4] -3, z[5] - 0, z[6] -2, z[7])
