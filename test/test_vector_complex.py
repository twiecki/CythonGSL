import unittest, vector_complex, os

class Block_complexTest(unittest.TestCase):

    def test_gsl_block_complex(self):
        t = vector_complex.t_gsl_block_complex()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_vector_complex_set(self):
        t = vector_complex.t_gsl_vector_complex_set()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_vector_complex_fprintf(self):
        t = vector_complex.t_gsl_vector_complex_fprintf()
        self.assertEqual(t,True)
        os.unlink('test.dat')

    def test_gsl_vector_complex_set_zero(self):
        t = vector_complex.t_gsl_vector_complex_set_zero()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_vector_complex_set_all(self):
        t = vector_complex.t_gsl_vector_complex_set_all()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_vector_complex_set_basis(self):
        t = vector_complex.t_gsl_vector_complex_set_basis()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_vector_complex_memcpy(self):
        t = vector_complex.t_gsl_vector_complex_memcpy()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_vector_complex_reverse(self):
        t = vector_complex.t_gsl_vector_complex_reverse()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_vector_complex_swap(self):
        t = vector_complex.t_gsl_vector_complex_swap()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_vector_complex_swap_elements(self):
        t = vector_complex.t_gsl_vector_complex_swap_elements()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_vector_complex_real(self):
        t = vector_complex.t_gsl_vector_complex_real()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_vector_complex_isnull(self):
        t = vector_complex.t_gsl_vector_complex_isnull()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_vector_complex_subvector(self):
        t = vector_complex.t_gsl_vector_complex_subvector()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_vector_complex_subvector_with_stride(self):
        t = vector_complex.t_gsl_vector_complex_subvector_with_stride()
        for x in t:
            self.assertAlmostEqual(x,0, 14)

    def test_gsl_vector_complex_view_array(self):
        t = vector_complex.t_gsl_vector_complex_view_array()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

if __name__ == '__main__':
    unittest.main()
