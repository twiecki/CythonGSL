from cython_gsl cimport *

def main ():
    cdef int i
    cdef double xi, yi, x[10], y[10]

    print "#m=0,S=2\n"

    for i  from 0 <= i < 10:
        x[i] = i + 0.5 * sin (i)
        y[i] = i + cos (i * i)
        print "%g %g\n" %(x[i], y[i])

    print "#m=1,S=0\n"

    cdef gsl_interp_accel *acc
    acc = gsl_interp_accel_alloc ()
    cdef gsl_spline *spline
    spline = gsl_spline_alloc (gsl_interp_cspline, 10)

    gsl_spline_init (spline, x, y, 10)

    xi = x[0]
    while (xi < x[9]):
        yi = gsl_spline_eval (spline, xi, acc)
        print "%g %g\n" %(xi, yi)
        xi = xi + 0.01

    gsl_spline_free (spline)
    gsl_interp_accel_free (acc)
