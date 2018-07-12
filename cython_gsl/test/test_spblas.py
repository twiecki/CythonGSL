import unittest, spblas

class SpblasTest(unittest.TestCase):
	def test_gsl_spblas_dgemm(self):
		t = spblas.t_gsl_spblas_dgemm()
		expected = [[0.0, 0.0, 0.0, 4.0], [0.0, 0.0, 15.0, 18.0]]
		self.assertListEqual(t[0], expected[0])
		self.assertListEqual(t[1], expected[1])

	def test_gsl_spblas_dgemv(self):
		t = spblas.t_gsl_spblas_dgemv()
		expected = [16.0, 15.0]
		self.assertListEqual(t, expected)

if __name__ == '__main__':
    unittest.main()
