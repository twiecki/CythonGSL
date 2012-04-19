import unittest, matrix, os

class MatrixTest(unittest.TestCase):

    def test_gsl_matrix_alloc(self):
        t = matrix.t_gsl_matrix_alloc()
        for x in t:
            self.assertAlmostEqual(x,0, 13)

    def test_gsl_matrix_calloc(self):
        t = matrix.t_gsl_matrix_calloc()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_matrix_max(self):
        t = matrix.t_gsl_matrix_max()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_matrix_isnull(self):
        t = matrix.t_gsl_matrix_isnull()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_matrix_add(self):
        t = matrix.t_gsl_matrix_add()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_matrix_mul_elements(self):
        t = matrix.t_gsl_matrix_mul_elements()
        for x in t:
            self.assertAlmostEqual(x,0, 14)

    def test_gsl_matrix_scale(self):
        t = matrix.t_gsl_matrix_scale()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_matrix_add_constant(self):
        t = matrix.t_gsl_matrix_add_constant()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_matrix_add_diagonal(self):
        t = matrix.t_gsl_matrix_add_diagonal()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_matrix_memcpy(self):
        t = matrix.t_gsl_matrix_memcpy()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_matrix_swap(self):
        t = matrix.t_gsl_matrix_swap()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_matrix_swap_rows(self):
        t = matrix.t_gsl_matrix_swap_rows()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_matrix_swap_columns(self):
        t = matrix.t_gsl_matrix_swap_columns()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_matrix_set_identity(self):
        t = matrix.t_gsl_matrix_set_identity()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_matrix_transpose(self):
        t = matrix.t_gsl_matrix_transpose()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_matrix_transpose_memcpy(self):
        t = matrix.t_gsl_matrix_transpose_memcpy()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_matrix_row(self):
        t = matrix.t_gsl_matrix_row()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_matrix_column(self):
        t = matrix.t_gsl_matrix_column()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_matrix_submatrix(self):
        t = matrix.t_gsl_matrix_submatrix()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_matrix_diagonal(self):
        t = matrix.t_gsl_matrix_diagonal()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_matrix_subdiagonal(self):
        t = matrix.t_gsl_matrix_subdiagonal()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_matrix_superdiagonal(self):
        t = matrix.t_gsl_matrix_superdiagonal()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_matrix_view_array(self):
        t = matrix.t_gsl_matrix_view_array()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_matrix_view_vector(self):
        t = matrix.t_gsl_matrix_view_vector()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_matrix_fprintf(self):
        t = matrix.t_gsl_matrix_fprintf()
        os.unlink('test.dat')
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_matrix_set_row(self):
        t = matrix.t_gsl_matrix_set_row()
        for x in t:
            self.assertAlmostEqual(x,0, 13)

if __name__ == '__main__':
    unittest.main()
