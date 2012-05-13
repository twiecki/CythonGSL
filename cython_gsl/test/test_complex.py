import unittest, complex


class ComplexTest(unittest.TestCase):

    def test_gsl_complex_rect(self):
        t = complex.t_gsl_complex_rect()
        self.assertAlmostEqual(t[0],1.1, 14)
        self.assertAlmostEqual(t[1],2.2, 14)

    def test_gsl_complex_polar(self):
        t = complex.t_gsl_complex_polar()
        self.assertAlmostEqual(t[0],-1.2, 14)
        self.assertAlmostEqual(t[1], 0, 14)

    def test_GSL_SET_COMPLEX(self):
        t = complex.t_GSL_SET_COMPLEX()
        self.assertAlmostEqual(t[0],1.1, 14)


    def test_GSL_REAL(self):
        self.assertAlmostEqual(complex.t_GSL_REAL(),0, 14)

    def test_GSL_IMAG(self):
        self.assertAlmostEqual(complex.t_GSL_IMAG(),0, 14)


    def test_GSL_COMPLEX_ONE(self):
        t = complex.t_GSL_COMPLEX_ONE()
        self.assertAlmostEqual(t[0],1, 14)
        self.assertAlmostEqual(t[1],0, 14)

    def test_GSL_COMPLEX_ZERO(self):
        t = complex.t_GSL_COMPLEX_ZERO()
        self.assertAlmostEqual(t[0],0, 14)
        self.assertAlmostEqual(t[1],0, 14)

    def test_GSL_COMPLEX_NEGONE(self):
        t = complex.t_GSL_COMPLEX_NEGONE()
        self.assertAlmostEqual(t[0],-1, 14)
        self.assertAlmostEqual(t[1],0, 14)

    def test_GSL_COMPLEX_EQ(self):
        self.assertEqual(complex.t_GSL_COMPLEX_EQ(),1)


    def test_GSL_SET_REAL(self):
        t = complex.t_GSL_SET_REAL()
        self.assertAlmostEqual(t[0],0, 14)
        self.assertAlmostEqual(t[1],0, 14)

    def test_GSL_SET_IMAG(self):
        t = complex.t_GSL_SET_IMAG()
        self.assertAlmostEqual(t[0],0, 14)
        self.assertAlmostEqual(t[1],0, 14)

    def test_gsl_complex_arg(self):
        self.assertAlmostEqual(complex.t_gsl_complex_arg(),0, 14)

    def test_gsl_complex_abs(self):
        self.assertAlmostEqual(complex.t_gsl_complex_abs(),0, 14)

    def test_gsl_complex_abs2(self):
        self.assertAlmostEqual(complex.t_gsl_complex_abs2(),0, 14)

    def test_gsl_complex_logabs(self):
        self.assertAlmostEqual(complex.t_gsl_complex_logabs(),0, 14)

    def test_gsl_complex_add(self):
        t = complex.t_gsl_complex_add()
        self.assertAlmostEqual(t[0],0, 14)
        self.assertAlmostEqual(t[1],0, 14)

    def test_gsl_complex_sub(self):
        t = complex.t_gsl_complex_sub()
        self.assertAlmostEqual(t[0],0, 14)
        self.assertAlmostEqual(t[1],0, 14)

    def test_gsl_complex_mul(self):
        t = complex.t_gsl_complex_mul()
        self.assertAlmostEqual(t[0],0, 14)
        self.assertAlmostEqual(t[1],0, 14)

    def test_gsl_complex_div(self):
        t = complex.t_gsl_complex_div()
        self.assertAlmostEqual(t[0],0, 14)
        self.assertAlmostEqual(t[1],0, 14)

    def test_gsl_complex_add_real(self):
        t = complex.t_gsl_complex_add_real()
        self.assertAlmostEqual(t[0],0, 14)
        self.assertAlmostEqual(t[1],0, 14)

    def test_gsl_complex_sub_real(self):
        t = complex.t_gsl_complex_sub_real()
        self.assertAlmostEqual(t[0],0, 14)
        self.assertAlmostEqual(t[1],0, 14)

    def test_gsl_complex_mul_real(self):
        t = complex.t_gsl_complex_mul_real()
        self.assertAlmostEqual(t[0],0, 14)
        self.assertAlmostEqual(t[1],0, 14)

    def test_gsl_complex_div_real(self):
        t = complex.t_gsl_complex_div_real()
        self.assertAlmostEqual(t[0],0, 14)
        self.assertAlmostEqual(t[1],0, 14)

    def test_gsl_complex_add_imag(self):
        t = complex.t_gsl_complex_add_imag()
        self.assertAlmostEqual(t[0],0, 14)
        self.assertAlmostEqual(t[1],0, 14)

    def test_gsl_complex_sub_imag(self):
        t = complex.t_gsl_complex_sub_imag()
        self.assertAlmostEqual(t[0],0, 14)
        self.assertAlmostEqual(t[1],0, 14)

    def test_gsl_complex_mul_imag(self):
        t = complex.t_gsl_complex_mul_imag()
        self.assertAlmostEqual(t[0],0, 14)
        self.assertAlmostEqual(t[1],0, 14)

    def test_gsl_complex_div_imag(self):
        t = complex.t_gsl_complex_div_imag()
        self.assertAlmostEqual(t[0],0, 14)
        self.assertAlmostEqual(t[1],0, 14)

    def test_gsl_complex_conjugate(self):
        t = complex.t_gsl_complex_conjugate()
        self.assertAlmostEqual(t[0],0, 14)
        self.assertAlmostEqual(t[1],0, 14)

    def test_gsl_complex_inverse(self):
        t = complex.t_gsl_complex_inverse()
        self.assertAlmostEqual(t[0],0, 14)
        self.assertAlmostEqual(t[1],0, 14)

    def test_gsl_complex_negative(self):
        t = complex.t_gsl_complex_negative()
        self.assertAlmostEqual(t[0],0, 14)
        self.assertAlmostEqual(t[1],0, 14)

    def test_gsl_complex_sqrt(self):
        t = complex.t_gsl_complex_sqrt()
        self.assertAlmostEqual(t[0],0, 14)
        self.assertAlmostEqual(t[1],0, 14)

    def test_gsl_complex_sqrt_real(self):
        t = complex.t_gsl_complex_sqrt_real()
        self.assertAlmostEqual(t[0],0, 14)
        self.assertAlmostEqual(t[1],0, 14)

    def test_gsl_complex_pow(self):
        t = complex.t_gsl_complex_pow()
        self.assertAlmostEqual(t[0],0, 14)
        self.assertAlmostEqual(t[1],0, 14)

    def test_gsl_complex_pow_real(self):
        t = complex.t_gsl_complex_pow_real()
        self.assertAlmostEqual(t[0],0, 14)
        self.assertAlmostEqual(t[1],0, 14)

    def test_gsl_complex_exp(self):
        t = complex.t_gsl_complex_exp()
        self.assertAlmostEqual(t[0],0, 14)
        self.assertAlmostEqual(t[1],0, 14)

    def test_gsl_complex_log(self):
        t = complex.t_gsl_complex_log()
        self.assertAlmostEqual(t[0],0, 14)
        self.assertAlmostEqual(t[1],0, 14)

    def test_gsl_complex_log10(self):
        t = complex.t_gsl_complex_log10()
        self.assertAlmostEqual(t[0],0, 14)
        self.assertAlmostEqual(t[1],0, 14)

    def test_gsl_complex_log_b(self):
        t = complex.t_gsl_complex_log_b()
        self.assertAlmostEqual(t[0],0, 14)
        self.assertAlmostEqual(t[1],0, 14)

    def test_gsl_complex_sin(self):
        t = complex.t_gsl_complex_sin()
        self.assertAlmostEqual(t[0],0, 14)
        self.assertAlmostEqual(t[1],0, 14)

    def test_gsl_complex_cos(self):
        t = complex.t_gsl_complex_cos()
        self.assertAlmostEqual(t[0],0, 14)
        self.assertAlmostEqual(t[1],0, 14)

    def test_gsl_complex_tan(self):
        t = complex.t_gsl_complex_tan()
        self.assertAlmostEqual(t[0],0, 14)
        self.assertAlmostEqual(t[1],0, 14)

    def test_gsl_complex_sec(self):
        t = complex.t_gsl_complex_sec()
        self.assertAlmostEqual(t[0],0, 14)
        self.assertAlmostEqual(t[1],0, 14)

    def test_gsl_complex_csc(self):
        t = complex.t_gsl_complex_csc()
        self.assertAlmostEqual(t[0],0, 14)
        self.assertAlmostEqual(t[1],0, 14)

    def test_gsl_complex_cot(self):
        t = complex.t_gsl_complex_cot()
        self.assertAlmostEqual(t[0],0, 14)
        self.assertAlmostEqual(t[1],0, 14)

    def test_gsl_complex_arcsin(self):
        t = complex.t_gsl_complex_arcsin()
        self.assertAlmostEqual(t[0],0, 14)
        self.assertAlmostEqual(t[1],0, 14)

    def test_gsl_complex_arcsin_real(self):
        t = complex.t_gsl_complex_arcsin_real()
        self.assertAlmostEqual(t[0],0, 14)
        self.assertAlmostEqual(t[1],0, 14)

    def test_gsl_complex_arccos(self):
        t = complex.t_gsl_complex_arccos()
        self.assertAlmostEqual(t[0],0, 14)
        self.assertAlmostEqual(t[1],0, 14)

    def test_gsl_complex_arccos_real(self):
        t = complex.t_gsl_complex_arccos_real()
        self.assertAlmostEqual(t[0],0, 14)
        self.assertAlmostEqual(t[1],0, 14)


    def test_gsl_complex_arctan(self):
        t = complex.t_gsl_complex_arctan()
        self.assertAlmostEqual(t[0],0, 14)
        self.assertAlmostEqual(t[1],0, 14)

    def test_gsl_complex_arcsec(self):
        t = complex.t_gsl_complex_arcsec()
        self.assertAlmostEqual(t[0],0, 14)
        self.assertAlmostEqual(t[1],0, 14)

    def test_gsl_complex_arcsec_real(self):
        t = complex.t_gsl_complex_arcsec_real()
        self.assertAlmostEqual(t[0],0, 14)
        self.assertAlmostEqual(t[1],0, 14)

    def test_gsl_complex_arccsc(self):
        t = complex.t_gsl_complex_arccsc()
        self.assertAlmostEqual(t[0],0, 14)
        self.assertAlmostEqual(t[1],0, 14)

    def test_gsl_complex_arccsc_real(self):
        t = complex.t_gsl_complex_arccsc_real()
        self.assertAlmostEqual(t[0],0, 14)
        self.assertAlmostEqual(t[1],0, 14)

    def test_gsl_complex_arccot(self):
        t = complex.t_gsl_complex_arccot()
        self.assertAlmostEqual(t[0],0, 14)
        self.assertAlmostEqual(t[1],0, 14)

    def test_gsl_complex_sinh(self):
        t = complex.t_gsl_complex_sinh()
        self.assertAlmostEqual(t[0],0, 14)
        self.assertAlmostEqual(t[1],0, 14)

    def test_gsl_complex_cosh(self):
        t = complex.t_gsl_complex_cosh()
        self.assertAlmostEqual(t[0],0, 14)
        self.assertAlmostEqual(t[1],0, 14)

    def test_gsl_complex_tanh(self):
        t = complex.t_gsl_complex_tanh()
        self.assertAlmostEqual(t[0],0, 14)
        self.assertAlmostEqual(t[1],0, 14)

    def test_gsl_complex_sech(self):
        t = complex.t_gsl_complex_sech()
        self.assertAlmostEqual(t[0],0, 14)
        self.assertAlmostEqual(t[1],0, 14)

    def test_gsl_complex_csch(self):
        t = complex.t_gsl_complex_csch()
        self.assertAlmostEqual(t[0],0, 14)
        self.assertAlmostEqual(t[1],0, 14)

    def test_gsl_complex_coth(self):
        t = complex.t_gsl_complex_coth()
        self.assertAlmostEqual(t[0],0, 14)
        self.assertAlmostEqual(t[1],0, 14)

    def test_gsl_complex_arcsinh(self):
        t = complex.t_gsl_complex_arcsinh()
        self.assertAlmostEqual(t[0],0, 14)
        self.assertAlmostEqual(t[1],0, 14)

    def test_gsl_complex_arccosh(self):
        t = complex.t_gsl_complex_arccosh()
        self.assertAlmostEqual(t[0],0, 14)
        self.assertAlmostEqual(t[1],0, 14)

    def test_gsl_complex_arccosh_real(self):
        t = complex.t_gsl_complex_arccosh_real()
        self.assertAlmostEqual(t[0],0, 14)
        self.assertAlmostEqual(t[1],0, 14)

    def test_gsl_complex_arctanh(self):
        t = complex.t_gsl_complex_arctanh()
        self.assertAlmostEqual(t[0],0, 14)
        self.assertAlmostEqual(t[1],0, 14)

    def test_gsl_complex_arctanh_real(self):
        t = complex.t_gsl_complex_arctanh_real()
        self.assertAlmostEqual(t[0],0, 14)
        self.assertAlmostEqual(t[1],0, 14)

    def test_gsl_complex_arcsech(self):
        t = complex.t_gsl_complex_arcsech()
        self.assertAlmostEqual(t[0],0, 14)
        self.assertAlmostEqual(t[1],0, 14)

    def test_gsl_complex_arccsch(self):
        t = complex.t_gsl_complex_arccsch()
        self.assertAlmostEqual(t[0],0, 14)
        self.assertAlmostEqual(t[1],0, 14)

    def test_gsl_complex_arccoth(self):
        t = complex.t_gsl_complex_arccoth()
        self.assertAlmostEqual(t[0],0, 14)
        self.assertAlmostEqual(t[1],0, 14)

if __name__ == '__main__':
    unittest.main()
