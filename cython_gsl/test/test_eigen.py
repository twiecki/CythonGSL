import unittest, eigen, os

class EigenTest(unittest.TestCase):

    def test_gsl_eigen_symm(self):
        t = eigen.t_gsl_eigen_symm()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_eigen_symmv(self):
        t = eigen.t_gsl_eigen_symmv()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_eigen_herm(self):
        t = eigen.t_gsl_eigen_herm()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_eigen_hermv(self):
        t = eigen.t_gsl_eigen_hermv()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

if __name__ == '__main__':
    unittest.main()
