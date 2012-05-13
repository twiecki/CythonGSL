from cython_gsl cimport *

def t_gsl_complex_rect():
    cdef double a, b, a1, b1
    a = 1.1
    b = 2.2
    a1 = GSL_REAL(gsl_complex_rect(a, b))
    b1 = GSL_IMAG(gsl_complex_rect(a, b))
    return (a1, b1)

def t_gsl_complex_polar():
    cdef gsl_complex z1
    z1 = gsl_complex_polar(1.2, M_PI)
    return (GSL_REAL(z1), GSL_IMAG (z1))


def t_GSL_SET_COMPLEX():
    cdef gsl_complex z1
    GSL_SET_COMPLEX(&z1,1.1, 2.2)
    return (GSL_REAL(z1), GSL_IMAG (z1))

def t_GSL_REAL():
    cdef gsl_complex z1
    GSL_SET_COMPLEX(&z1,1.1, 2.2)
    return GSL_REAL(z1) - 1.1

def t_GSL_IMAG():
    cdef gsl_complex z1
    GSL_SET_COMPLEX(&z1,1.1, 2.2)
    return GSL_IMAG(z1) - 2.2


def t_GSL_COMPLEX_ONE():
    cdef gsl_complex z
    z = GSL_COMPLEX_ONE
    return (GSL_REAL(z), GSL_IMAG(z))

def t_GSL_COMPLEX_ZERO():
    return (GSL_REAL(GSL_COMPLEX_ZERO), GSL_IMAG(GSL_COMPLEX_ZERO))

def t_GSL_COMPLEX_NEGONE():
    return (GSL_REAL(GSL_COMPLEX_NEGONE), GSL_IMAG(GSL_COMPLEX_NEGONE))

def t_GSL_COMPLEX_EQ():
    cdef gsl_complex z1, z2
    GSL_SET_COMPLEX(&z1,1.1, 2.2)
    GSL_SET_COMPLEX(&z2,1.1, 2.2)
    return GSL_COMPLEX_EQ(z1,z2)


def t_GSL_SET_REAL():
    cdef gsl_complex z1
    GSL_SET_COMPLEX(&z1,1.1, 2.2)
    GSL_SET_REAL(&z1, 3.2)
    return (GSL_REAL(z1) -3.2, GSL_IMAG(z1) -2.2)

def t_GSL_SET_IMAG():
    cdef gsl_complex z1
    GSL_SET_COMPLEX(&z1,1.1, 2.2)
    GSL_SET_IMAG(&z1, 3.2)
    return (GSL_REAL(z1) -1.1, GSL_IMAG(z1) -3.2)

def t_gsl_complex_arg():
    cdef gsl_complex z
    GSL_SET_COMPLEX(&z,1.1, 2.2)
    return gsl_complex_arg(z) - atan(2)

def t_gsl_complex_abs():
    cdef gsl_complex z
    GSL_SET_COMPLEX(&z,1, 2)
    return gsl_complex_abs(z) - sqrt(5)

def t_gsl_complex_abs2():
    cdef gsl_complex z
    GSL_SET_COMPLEX(&z,1, 2)
    return gsl_complex_abs2(z) - 5

def t_gsl_complex_logabs():
    cdef gsl_complex z
    GSL_SET_COMPLEX(&z,1, 2)
    return gsl_complex_logabs(z) - log(5)/2

def t_gsl_complex_add():
    cdef gsl_complex z1, z2,z
    GSL_SET_COMPLEX(&z1,1, 2)
    GSL_SET_COMPLEX(&z2,2, 3)
    z = gsl_complex_add(z1,z2)
    return (GSL_REAL(z) - 3,GSL_IMAG(z) -5)

def t_gsl_complex_sub():
    cdef gsl_complex z1, z2,z
    GSL_SET_COMPLEX(&z1,1, 2)
    GSL_SET_COMPLEX(&z2,2, 3)
    z = gsl_complex_sub(z1,z2)
    return (GSL_REAL(z) + 1,GSL_IMAG(z) + 1 )

def t_gsl_complex_mul():
    cdef gsl_complex z1, z2,z
    GSL_SET_COMPLEX(&z1,1, 2)
    GSL_SET_COMPLEX(&z2,2, 3)
    z = gsl_complex_mul(z1,z2)
    return (GSL_REAL(z) + 4,GSL_IMAG(z) -7)

def t_gsl_complex_div():
    cdef gsl_complex z1, z2,z
    GSL_SET_COMPLEX(&z1,1, 2)
    GSL_SET_COMPLEX(&z2,2, 3)
    z = gsl_complex_div(z1,z2)
    return (GSL_REAL(z) - 8.0/13,GSL_IMAG(z) - 1.0/13)

def t_gsl_complex_add_real():
    cdef gsl_complex z1, z
    GSL_SET_COMPLEX(&z1,1.1, 2.3)
    cdef double a
    a = 2.0
    z = gsl_complex_add_real(z1, a)
    return (GSL_REAL(z) - 3.1,GSL_IMAG(z) - 2.3)

def t_gsl_complex_sub_real():
    cdef gsl_complex z1, z
    GSL_SET_COMPLEX(&z1,1.1, 2.3)
    cdef double a
    a = 2.0
    z = gsl_complex_sub_real(z1, a)
    return (GSL_REAL(z) + 0.9,GSL_IMAG(z) - 2.3)

def t_gsl_complex_mul_real():
    cdef gsl_complex z1, z
    GSL_SET_COMPLEX(&z1,1.1, 2.3)
    cdef double a
    a = 2.0
    z = gsl_complex_mul_real(z1, a)
    return (GSL_REAL(z) - 2.2,GSL_IMAG(z) - 4.6)

def t_gsl_complex_div_real():
    cdef gsl_complex z1, z
    GSL_SET_COMPLEX(&z1,1.1, 2.3)
    cdef double a
    a = 2.0
    z = gsl_complex_div_real(z1, a)
    return (GSL_REAL(z) - 0.55,GSL_IMAG(z) - 1.15)

def t_gsl_complex_add_imag():
    cdef gsl_complex z1, z
    GSL_SET_COMPLEX(&z1,1.1, 2.3)
    cdef double a
    a = 2.0
    z = gsl_complex_add_imag(z1, a)
    return (GSL_REAL(z) - 1.1,GSL_IMAG(z) - 4.3)

def t_gsl_complex_sub_imag():
    cdef gsl_complex z1, z
    GSL_SET_COMPLEX(&z1,1.1, 2.3)
    cdef double a
    a = 2.0
    z = gsl_complex_sub_imag(z1, a)
    return (GSL_REAL(z) - 1.1,GSL_IMAG(z) - 0.3)

def t_gsl_complex_mul_imag():
    cdef gsl_complex z1, z
    GSL_SET_COMPLEX(&z1,1.1, 2.3)
    cdef double a
    a = 2.0
    z = gsl_complex_mul_imag(z1, a)
    return (GSL_REAL(z) + 4.6,GSL_IMAG(z) - 2.2)

def t_gsl_complex_div_imag():
    cdef gsl_complex z1, z
    GSL_SET_COMPLEX(&z1,1.1, 2.3)
    cdef double a
    a = 2.0
    z = gsl_complex_div_imag(z1, a)
    return (GSL_REAL(z) - 1.15,GSL_IMAG(z) + 0.55)

def t_gsl_complex_conjugate():
    cdef gsl_complex z1, z
    GSL_SET_COMPLEX(&z1,1.1, 2.3)
    z = gsl_complex_conjugate(z1)
    return (GSL_REAL(z) - 1.1,GSL_IMAG(z) + 2.3)

def t_gsl_complex_inverse():
    cdef gsl_complex z1, z
    GSL_SET_COMPLEX(&z1,1, 1)
    z = gsl_complex_inverse(z1)
    return (GSL_REAL(z) - 1.0/2,GSL_IMAG(z) + 1.0/2)

def t_gsl_complex_negative():
    cdef gsl_complex z1, z
    GSL_SET_COMPLEX(&z1,1.1, 2.3)
    z = gsl_complex_negative(z1)
    return (GSL_REAL(z) + 1.1,GSL_IMAG(z) + 2.3)

def t_gsl_complex_sqrt():
    cdef gsl_complex z1, z
    z1 = gsl_complex_polar(4,0.4)
    z = gsl_complex_sqrt(z1)
    return (gsl_complex_abs(z) - 2, gsl_complex_arg(z) - 0.2)

def t_gsl_complex_sqrt_real():
    cdef gsl_complex z
    z = gsl_complex_sqrt_real(-1.0)
    return (GSL_REAL(z) ,GSL_IMAG(z) - 1)

def t_gsl_complex_pow():
    cdef gsl_complex z1, z, I
    z1 = gsl_complex_polar(1,0.4)
    GSL_SET_COMPLEX(&I, 0, 1)
    z = gsl_complex_pow(z1, I)
    return (GSL_REAL(z) - exp(-0.4), GSL_IMAG(z) )

def t_gsl_complex_pow_real():
    cdef gsl_complex z1, z
    z1 = gsl_complex_polar(2,0.4)
    z = gsl_complex_pow_real(z1, 2)
    return (gsl_complex_abs(z) - 4, gsl_complex_arg(z) - 0.8)

def t_gsl_complex_exp():
    cdef gsl_complex z1, z
    z1 = gsl_complex_rect(0, 0.4)
    z = gsl_complex_exp(z1)
    return (gsl_complex_abs(z) - 1, gsl_complex_arg(z) - 0.4)

def t_gsl_complex_log():
    cdef gsl_complex z1, z
    z1 = gsl_complex_polar(1,0.4)
    z = gsl_complex_log(z1)
    return (GSL_REAL(z), GSL_IMAG(z) - 0.4)


def t_gsl_complex_log10():
    cdef gsl_complex z1, z2, z3, z
    GSL_SET_COMPLEX(&z1,10,0)
    GSL_SET_COMPLEX(&z2, 2,1)
    z3 = gsl_complex_pow(z1, z2)
    z = gsl_complex_log10(z3)
    return (GSL_REAL(z) - 2, GSL_IMAG(z) - 1 )

def t_gsl_complex_log_b():
    cdef gsl_complex z1, z2, z3, z
    GSL_SET_COMPLEX(&z1,10,2)
    GSL_SET_COMPLEX(&z2, 2,1)
    z3 = gsl_complex_pow(z1, z2)
    z = gsl_complex_log_b(z3, z1)
    return (GSL_REAL(z) - 2, GSL_IMAG(z) - 1 )

def t_gsl_complex_sin():
    cdef gsl_complex z1, z
    GSL_SET_COMPLEX(&z1,2.1,0)
    z = gsl_complex_sin(z1)
    return (GSL_REAL(z) - sin(2.1), GSL_IMAG(z) )

def t_gsl_complex_cos():
    cdef gsl_complex z1, z
    GSL_SET_COMPLEX(&z1,2.1,0)
    z = gsl_complex_cos(z1)
    return (GSL_REAL(z) - cos(2.1), GSL_IMAG(z) )

def t_gsl_complex_tan():
    cdef gsl_complex z1, z
    GSL_SET_COMPLEX(&z1,2.1,0)
    z = gsl_complex_tan(z1)
    return (GSL_REAL(z) - tan(2.1), GSL_IMAG(z) )

def t_gsl_complex_sec():
    cdef gsl_complex z1, z2, z
    GSL_SET_COMPLEX(&z1,2.1, 3.2)
    z2 = gsl_complex_div(GSL_COMPLEX_ONE,gsl_complex_cos(z1))
    z1 = gsl_complex_sec(z1)
    z = gsl_complex_sub(z1,z2)
    return (GSL_REAL(z), GSL_IMAG(z))

def t_gsl_complex_csc():
    cdef gsl_complex z1, z2, z
    GSL_SET_COMPLEX(&z1,2.1, 3.2)
    z2 = gsl_complex_div(GSL_COMPLEX_ONE,gsl_complex_sin(z1))
    z1 = gsl_complex_csc(z1)
    z = gsl_complex_sub(z1,z2)
    return (GSL_REAL(z), GSL_IMAG(z))

def t_gsl_complex_cot():
    cdef gsl_complex z1, z2, z
    GSL_SET_COMPLEX(&z1,2.1, 3.2)
    z2 = gsl_complex_div(GSL_COMPLEX_ONE,gsl_complex_tan(z1))
    z1 = gsl_complex_cot(z1)
    z = gsl_complex_sub(z1,z2)
    return (GSL_REAL(z), GSL_IMAG(z))

def t_gsl_complex_arcsin():
    cdef gsl_complex z1, z
    GSL_SET_COMPLEX(&z1,2.1,3)
    z = gsl_complex_arcsin(z1)
    z = gsl_complex_sin(z)
    return (GSL_REAL(z) - 2.1, GSL_IMAG(z) -3 )

def t_gsl_complex_arcsin_real():
    cdef gsl_complex z1
    z1 = gsl_complex_arcsin_real(0.5)
    return (GSL_REAL(z1) - M_PI/6, GSL_IMAG(z1) )


def t_gsl_complex_arccos():
    cdef gsl_complex z1, z
    GSL_SET_COMPLEX(&z1,2.1,2.2)
    z = gsl_complex_arccos(z1)
    z = gsl_complex_cos(z)
    return (GSL_REAL(z) - 2.1, GSL_IMAG(z) - 2.2 )

def t_gsl_complex_arccos_real():
    cdef gsl_complex z1
    z1 = gsl_complex_arccos_real(0.5)
    return (GSL_REAL(z1) - M_PI/3, GSL_IMAG(z1) )


def t_gsl_complex_arctan():
    cdef gsl_complex z1, z
    GSL_SET_COMPLEX(&z1,2.1,3)
    z = gsl_complex_arctan(z1)
    z = gsl_complex_tan(z)
    return (GSL_REAL(z) - 2.1, GSL_IMAG(z) -3 )

def t_gsl_complex_arcsec():
    cdef gsl_complex z1, z
    GSL_SET_COMPLEX(&z1,2.1,3)
    z = gsl_complex_arcsec(z1)
    z = gsl_complex_sec(z)
    return (GSL_REAL(z) - 2.1, GSL_IMAG(z) -3 )

def t_gsl_complex_arcsec_real():
    cdef gsl_complex z1, z
    z1 = gsl_complex_arcsec_real(2.5)
    z = gsl_complex_sec(z1)
    return (GSL_REAL(z) - 2.5, GSL_IMAG(z) )

def t_gsl_complex_arccsc():
    cdef gsl_complex z1, z
    GSL_SET_COMPLEX(&z1,2.1,3)
    z = gsl_complex_arccsc(z1)
    z = gsl_complex_csc(z)
    return (GSL_REAL(z) - 2.1, GSL_IMAG(z) -3 )

def t_gsl_complex_arccsc_real():
    cdef gsl_complex z1, z
    z1 = gsl_complex_arccsc_real(2.5)
    z = gsl_complex_csc(z1)
    return (GSL_REAL(z) - 2.5, GSL_IMAG(z) )

def t_gsl_complex_arccot():
    cdef gsl_complex z1, z
    GSL_SET_COMPLEX(&z1,2.1,3)
    z = gsl_complex_arccot(z1)
    z = gsl_complex_cot(z)
    return (GSL_REAL(z) - 2.1, GSL_IMAG(z) -3 )

def t_gsl_complex_sinh():
    cdef gsl_complex z1, z2, z3, z
    GSL_SET_COMPLEX(&z1,2.1,3)
    z2 = gsl_complex_exp(z1)
    z3 = gsl_complex_exp(gsl_complex_negative(z1))
    z = gsl_complex_sub(z2,z3)
    z = gsl_complex_div_real(z,2)
    z2 = gsl_complex_sinh(z1)
    z = gsl_complex_sub(z,z2)
    return (GSL_REAL(z), GSL_IMAG(z))

def t_gsl_complex_cosh():
    cdef gsl_complex z1, z2, z3, z
    GSL_SET_COMPLEX(&z1,2.1,3)
    z2 = gsl_complex_exp(z1)
    z3 = gsl_complex_exp(gsl_complex_negative(z1))
    z = gsl_complex_add(z2,z3)
    z = gsl_complex_div_real(z,2)
    z2 = gsl_complex_cosh(z1)
    z = gsl_complex_sub(z,z2)
    return (GSL_REAL(z), GSL_IMAG(z))

def t_gsl_complex_tanh():
    cdef gsl_complex z1, z2, z
    GSL_SET_COMPLEX(&z1,2.1,3)
    z2 = gsl_complex_div(gsl_complex_sinh(z1), gsl_complex_cosh(z1))
    z = gsl_complex_tanh(z1)
    z = gsl_complex_sub(z,z2)
    return (GSL_REAL(z), GSL_IMAG(z))

def t_gsl_complex_sech():
    cdef gsl_complex z1, z2, z
    GSL_SET_COMPLEX(&z1,2.1, 3.2)
    z2 = gsl_complex_div(GSL_COMPLEX_ONE,gsl_complex_cosh(z1))
    z1 = gsl_complex_sech(z1)
    z = gsl_complex_sub(z1,z2)
    return (GSL_REAL(z), GSL_IMAG(z))

def t_gsl_complex_csch():
    cdef gsl_complex z1, z2, z
    GSL_SET_COMPLEX(&z1,2.1, 3.2)
    z2 = gsl_complex_div(GSL_COMPLEX_ONE,gsl_complex_sinh(z1))
    z1 = gsl_complex_csch(z1)
    z = gsl_complex_sub(z1,z2)
    return (GSL_REAL(z), GSL_IMAG(z))

def t_gsl_complex_coth():
    cdef gsl_complex z1, z2, z
    GSL_SET_COMPLEX(&z1,2.1, 3.2)
    z2 = gsl_complex_div(GSL_COMPLEX_ONE,gsl_complex_tanh(z1))
    z1 = gsl_complex_coth(z1)
    z = gsl_complex_sub(z1,z2)
    return (GSL_REAL(z), GSL_IMAG(z))

def t_gsl_complex_arcsinh():
    cdef gsl_complex z1, z
    GSL_SET_COMPLEX(&z1,2.1,3)
    z = gsl_complex_arcsinh(z1)
    z = gsl_complex_sinh(z)
    return (GSL_REAL(z) - 2.1, GSL_IMAG(z) -3 )

def t_gsl_complex_arccosh():
    cdef gsl_complex z1, z
    GSL_SET_COMPLEX(&z1,2.1,3)
    z = gsl_complex_arccosh(z1)
    z = gsl_complex_cosh(z)
    return (GSL_REAL(z) - 2.1, GSL_IMAG(z) -3 )

def t_gsl_complex_arccosh_real():
    cdef gsl_complex z1, z
    z1 = gsl_complex_arccosh_real(2.5)
    z = gsl_complex_cosh(z1)
    return (GSL_REAL(z) - 2.5, GSL_IMAG(z) )

def t_gsl_complex_arctanh():
    cdef gsl_complex z1, z
    GSL_SET_COMPLEX(&z1,2.1,3)
    z = gsl_complex_arctanh(z1)
    z = gsl_complex_tanh(z)
    return (GSL_REAL(z) - 2.1, GSL_IMAG(z) -3 )

def t_gsl_complex_arctanh_real():
    cdef gsl_complex z1, z
    z1 = gsl_complex_arctanh_real(2.5)
    z = gsl_complex_tanh(z1)
    return (GSL_REAL(z) - 2.5, GSL_IMAG(z) )

def t_gsl_complex_arcsech():
    cdef gsl_complex z1, z
    GSL_SET_COMPLEX(&z1,2.1,3)
    z = gsl_complex_arcsech(z1)
    z = gsl_complex_sech(z)
    return (GSL_REAL(z) - 2.1, GSL_IMAG(z) -3 )

def t_gsl_complex_arccsch():
    cdef gsl_complex z1, z
    GSL_SET_COMPLEX(&z1,2.1,3)
    z = gsl_complex_arccsch(z1)
    z = gsl_complex_csch(z)
    return (GSL_REAL(z) - 2.1, GSL_IMAG(z) -3 )

def t_gsl_complex_arccoth():
    cdef gsl_complex z1, z
    GSL_SET_COMPLEX(&z1,2.1,3)
    z = gsl_complex_arccoth(z1)
    z = gsl_complex_coth(z)
    return (GSL_REAL(z) - 2.1, GSL_IMAG(z) -3 )
