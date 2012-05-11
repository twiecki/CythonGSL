import unittest, vector, os

class BlockTest(unittest.TestCase):
    def test_gsl_block(self):
        t = vector.t_gsl_block()
        for x in t:
            self.assertEqual(x, 0)

class VectorTest(unittest.TestCase):

    def test_gsl_vector_set(self):
        t = vector.t_gsl_vector_set()
        for x in t:
            self.assertAlmostEqual(x, 0, 15)

    def test_gsl_vector_fprintf(self):
        self.assertEqual(vector.t_gsl_vector_fprintf(), False)
        os.unlink("test.dat")

    def test_gsl_vector_set_zero(self):
        t = vector.t_gsl_vector_set_zero()
        for x in t:
            self.assertEqual(x,0)

    def test_gsl_vector_set_all(self):
        t = vector.t_gsl_vector_set_all()
        for x in t:
            self.assertEqual(x,0)

    def test_gsl_vector_set_basis(self):
        t = vector.t_gsl_vector_set_basis()
        for x in t:
            self.assertEqual(x,0)

    def test_gsl_vector_calloc(self):
        t = vector.t_gsl_vector_calloc()
        for x in t:
            self.assertEqual(x,0)

    def test_gsl_vector_memcpy(self):
        t = vector.t_gsl_vector_memcpy()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_vector_reverse(self):
        t = vector.t_gsl_vector_reverse()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_vector_swap(self):
        t = vector.t_gsl_vector_swap()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_vector_swap_elements(self):
        t = vector.t_gsl_vector_swap_elements()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_vector_max(self):
        t = vector.t_gsl_vector_max()
        for x in t:
            self.assertAlmostEqual(x,0, 15)


    def test_gsl_vector_max_index(self):
        t = vector.t_gsl_vector_max_index()
        for x in t:
            self.assertAlmostEqual(x,0, 15)


    def test_gsl_vector_add(self):
        t = vector.t_gsl_vector_add()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_vector_sub(self):
        t = vector.t_gsl_vector_sub()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_vector_mul(self):
        t = vector.t_gsl_vector_mul()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_vector_div(self):
        t = vector.t_gsl_vector_div()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_vector_scale(self):
        t = vector.t_gsl_vector_scale()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_vector_add_constant(self):
        t = vector.t_gsl_vector_add_constant()
        for x in t:
            self.assertAlmostEqual(x,0, 14)

    def test_gsl_vector_isnull(self):
        t = vector.t_gsl_vector_isnull()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_vector_subvector(self):
        t = vector.t_gsl_vector_subvector()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_vector_subvector_with_stride1(self):
        t = vector.t_gsl_vector_subvector_with_stride1()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_vector_subvector_with_stride2(self):
        t = vector.t_gsl_vector_subvector_with_stride2()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_vector_view_array(self):
        t = vector.t_gsl_vector_view_array()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_vector_view_array_with_stride(self):
        t = vector.t_gsl_vector_view_array_with_stride()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

if __name__ == '__main__':
    unittest.main()
