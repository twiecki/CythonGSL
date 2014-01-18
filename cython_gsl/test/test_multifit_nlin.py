import unittest, multifit_nlin


class MultifitNlinTest(unittest.TestCase):

    def test_multifit_nlin_example(self):
        A, lambd, b = multifit_nlin.t_gsl_multifit_nlin_example()
        self.assertAlmostEqual(A, 5.04536, 5)
        self.assertAlmostEqual(lambd, 0.10405, 5)
        self.assertAlmostEqual(b, 1.01925, 5)


if __name__ == '__main__':
    unittest.main()
