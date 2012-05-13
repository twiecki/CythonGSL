import unittest, matrix_complex, os

class MatrixComplexTest(unittest.TestCase):

    def test_gsl_matrix_complex_set(self):
        t = matrix_complex.t_gsl_matrix_complex_set()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_matrix_complex_calloc(self):
        t = matrix_complex.t_gsl_matrix_complex_calloc()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_matrix_complex_isnull(self):
        t = matrix_complex.t_gsl_matrix_complex_isnull()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_matrix_complex_add(self):
        t = matrix_complex.t_gsl_matrix_complex_add()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_matrix_complex_mul_elements(self):
        t = matrix_complex.t_gsl_matrix_complex_mul_elements()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_matrix_complex_scale(self):
        t = matrix_complex.t_gsl_matrix_complex_scale()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_matrix_complex_add_constant(self):
        t = matrix_complex.t_gsl_matrix_complex_add_constant()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_matrix_complex_add_diagonal(self):
        t = matrix_complex.t_gsl_matrix_complex_add_diagonal()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_matrix_complex_memcpy(self):
        t = matrix_complex.t_gsl_matrix_complex_memcpy()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_matrix_complex_swap(self):
        t = matrix_complex.t_gsl_matrix_complex_swap()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_matrix_complex_swap_rows(self):
        t = matrix_complex.t_gsl_matrix_complex_swap_rows()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_matrix_complex_swap_columns(self):
        t = matrix_complex.t_gsl_matrix_complex_swap_columns()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_matrix_complex_set_identity(self):
        t = matrix_complex.t_gsl_matrix_complex_set_identity()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_matrix_complex_transpose(self):
        t = matrix_complex.t_gsl_matrix_complex_transpose()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_matrix_complex_transpose_memcpy(self):
        t = matrix_complex.t_gsl_matrix_complex_transpose_memcpy()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_matrix_complex_row(self):
        t = matrix_complex.t_gsl_matrix_complex_row()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_matrix_complex_column(self):
        t = matrix_complex.t_gsl_matrix_complex_column()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_matrix_complex_submatrix(self):
        t = matrix_complex.t_gsl_matrix_complex_submatrix()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_matrix_complex_diagonal(self):
        t = matrix_complex.t_gsl_matrix_complex_diagonal()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_matrix_complex_subdiagonal(self):
        t = matrix_complex.t_gsl_matrix_complex_subdiagonal()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_matrix_complex_superdiagonal(self):
        t = matrix_complex.t_gsl_matrix_complex_superdiagonal()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_matrix_complex_view_array(self):
        t = matrix_complex.t_gsl_matrix_complex_view_array()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_matrix_complex_view_vector(self):
        t = matrix_complex.t_gsl_matrix_complex_view_vector()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_matrix_complex_fprintf(self):
        t = matrix_complex.t_gsl_matrix_complex_fprintf()
        os.unlink('test.dat')
        for x in t:
            self.assertAlmostEqual(x,0, 15)

if __name__ == '__main__':
    unittest.main()
