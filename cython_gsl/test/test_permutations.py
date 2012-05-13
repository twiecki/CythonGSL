import unittest, permutations, os

class PermutationTest(unittest.TestCase):

    def test_gsl_permutation_alloc(self):
        t = permutations.t_gsl_permutation_alloc()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_permutation_calloc(self):
        t = permutations.t_gsl_permutation_calloc()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_permutation_get(self):
        t = permutations.t_gsl_permutation_get()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_permutation_next(self):
        t = permutations.t_gsl_permutation_next()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_permutation_prev(self):
        t = permutations.t_gsl_permutation_prev()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_permutation_valid(self):
        t = permutations.t_gsl_permutation_valid()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_permutation_swap(self):
        t = permutations.t_gsl_permutation_swap()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_permutation_memcpy(self):
        t = permutations.t_gsl_permutation_memcpy()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_permutation_reverse(self):
        t = permutations.t_gsl_permutation_reverse()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_permutation_inverse(self):
        t = permutations.t_gsl_permutation_inverse()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_permute_vector(self):
        t = permutations.t_gsl_permute_vector()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_permute_vector_inverse(self):
        t = permutations.t_gsl_permute_vector_inverse()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_permute_vector_complex(self):
        t = permutations.t_gsl_permute_vector_complex()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_permute_vector_complex_inverse(self):
        t = permutations.t_gsl_permute_vector_complex_inverse()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_permutation_linear_to_canonical(self):
        t = permutations.t_gsl_permutation_linear_to_canonical()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_permutation_fprintf(self):
        t = permutations.t_gsl_permutation_fprintf()
        os.unlink('test.dat')
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_permutation_linear_cycles(self):
        t = permutations.t_gsl_permutation_linear_cycles()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

if __name__ == '__main__':
    unittest.main()
