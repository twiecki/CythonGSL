import unittest, poly


class PolyTest(unittest.TestCase):


    def test_gsl_poly_eval(self):
        t = poly.t_gsl_poly_eval()
        self.assertAlmostEqual(t,0, 15)

    def test_gsl_poly_dd_init(self):
        t = poly.t_gsl_poly_dd_init()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_poly_solve_quadratic(self):
        t = poly.t_gsl_poly_solve_quadratic()
        for x in t:
            self.assertAlmostEqual(x, 0, 15)

    def test_gsl_poly_complex_solve_quadratic(self):
        t = poly.t_gsl_poly_complex_solve_quadratic()
        for x in t:
            self.assertAlmostEqual(x, 0, 15)

    def test_gsl_poly_solve_cubic1(self):
        t = poly.t_gsl_poly_solve_cubic1()
        for x in t:
            self.assertAlmostEqual(x, 0, 15)

    def test_gsl_poly_solve_cubic2(self):
        t = poly.t_gsl_poly_solve_cubic2()
        for x in t:
            self.assertAlmostEqual(x, 0, 15)

    def test_gsl_poly_complex_solve_cubic1(self):
        t = poly.t_gsl_poly_complex_solve_cubic1()
        for x in t:
            self.assertAlmostEqual(x, 0, 15)

    def test_gsl_poly_complex_solve_cubic2(self):
        t = poly.t_gsl_poly_complex_solve_cubic2()
        for x in t:
            self.assertAlmostEqual(x, 0, 15)

    def test_gsl_poly_complex_solve(self):
        t = poly.t_gsl_poly_complex_solve()
        for x in t:
            self.assertAlmostEqual(x, 0, 14)

if __name__ == '__main__':
    unittest.main()
