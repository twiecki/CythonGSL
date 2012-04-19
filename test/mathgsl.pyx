#cython: cdivision=True

from cython_gsl cimport *


def t_M_PI():
    return M_PI

def t_M_E():
    return M_E

def t_M_LOG2E():
    return M_LOG2E


def t_M_LOG10E():
    return M_LOG10E


def t_M_LN2():
    return M_LN2

def t_M_PI_2():
    return M_PI_2

def t_M_PI_4():
    return M_PI_4

def t_M_1_PI():
    return M_1_PI

def t_M_2_PI():
    return M_2_PI

def t_M_2_SQRTPI():
    return M_2_SQRTPI

def t_M_SQRT2():
    return M_SQRT2

def t_M_SQRT1_2():
    return M_SQRT1_2

def t_M_LNPI():
    return M_LNPI - log(M_PI)

def t_M_EULER():
    return M_EULER

def t_isnan1():
    cdef double inf, nan
    inf = exp (1.0e10)
    nan = inf/inf
    return gsl_isnan(nan)

def t_isnan2():
    return gsl_isnan(1.3)

def t_isinf1():
    cdef double inf
    inf = exp (1.0e10)
    return gsl_isinf(inf)


def t_isinf2():
    cdef double inf
    inf = -exp (1.0e10)
    return gsl_isinf(inf)

def t_isinf3():
    cdef double a
    a = exp (2.3)
    return gsl_isinf(a)

def t_finite1():
    cdef double inf
    inf = exp (1.0e10)
    return gsl_finite(inf)

def t_finite2():
    cdef double a
    a = exp (2.3)
    return gsl_finite(a)

def t_gsl_log1p(double x):
    return gsl_log1p(x) - log(1 + x)

def t_gsl_expm1(double x):
    return gsl_expm1(x) - (exp(x) - 1)

def t_gsl_hypot(double x, double y):
    return gsl_hypot(x, y) - sqrt(x*x + y*y)

def t_gsl_acosh(double x):
    return cosh(gsl_acosh(x)) - x

def t_gsl_asinh(double x):
    return sinh(gsl_asinh(x)) - x

def t_gsl_atanh(double x):
    return tanh(gsl_atanh(x)) - x

def t_gsl_ldexp(double x):
    return gsl_ldexp(x, 3) - x * 2**3

def t_gsl_frexp():
    cdef double a
    cdef int e
    a = gsl_frexp(0.6 * 2**-15, &e)
    return (a,e)

# Small integer powers
def t_gsl_pow_2(double x):
    return gsl_pow_2(x) - x*x

def t_gsl_pow_3(double x):
    return gsl_pow_3(x) - x*x*x

def t_gsl_pow_4(double x):
    return gsl_pow_4(x) - x*x*x*x

def t_gsl_pow_5(double x):
    return gsl_pow_5(x) - x*x*x*x*x

def t_gsl_pow_6(double x):
    return gsl_pow_6(x) - x*x*x*x*x*x

def t_gsl_pow_7(double x):
    return gsl_pow_7(x) - x*x*x*x*x*x*x

def t_gsl_pow_8(double x):
    return gsl_pow_8(x) - x*x*x*x*x*x*x*x

def t_gsl_pow_9(double x):
    return gsl_pow_9(x) - x*x*x*x*x*x*x*x*x

#Testing the Sign of Numbers
def t_GSL_SIGN(double x):
    return GSL_SIGN(x)

#Testing for Odd and Even Numbers
def t_GSL_IS_ODD(int x):
    return GSL_IS_ODD(x)

def t_GSL_IS_EVEN(int x):
    return GSL_IS_EVEN(x)

#Maximum and Minimum functions
def t_GSL_MAX(double x, double y):
    return GSL_MAX(x,y)

def t_GSL_MIN(double x, double y):
    return GSL_MIN(x,y)

def t_GSL_MAX_DBL(double x, double y):
    return GSL_MAX_DBL(x,y)

def t_GSL_MIN_DBL(double x, double y):
    return GSL_MIN_DBL(x,y)

def t_GSL_MAX_INT(int a, int b):
    return GSL_MAX_INT(a,b)

def t_GSL_MIN_INT(int a, int b):
    return GSL_MIN_INT(a,b)


#Approximate Comparison of Floating Point Numbers
def t_gsl_fcmp1( ):
    eps = 1.0e-15
    x = 1.0e-12; y = x/8
    return gsl_fcmp(x,y,eps)

def t_gsl_fcmp2( ):
    eps = 1.0e-15
    # returns 0, i.e. 1 + cos(M_PI/2) =~ 1
    return gsl_fcmp(1 + cos(M_PI/2),1,eps)

def t_gsl_fcmp3( ):
    eps = 1.0e-15
    # returns 1; cos(M_PI/2) !~ 0
    return gsl_fcmp(cos(M_PI/2),0.0,eps)

# Definition of an arbitrary function with parameters
# see ../examples/integration.pyx

cdef double gsl_function_f1(double x, void *p) nogil:
    cdef double a, b
    a = (<double*> p)[0]
    return a*x

cdef double gsl_function_a
gsl_function_a = 3
cdef gsl_function F
F.function = &gsl_function_f1
F.params = &gsl_function_a

def t_gsl_function():
    cdef double x
    x = 7
    return F.function(x, F.params) - x*gsl_function_a

def t_GSL_FN_EVAL():
    cdef double x
    x = 7
    return GSL_FN_EVAL(&F, x) - x*gsl_function_a

# Definition of an arbitrary function returning two values, r1, r2
