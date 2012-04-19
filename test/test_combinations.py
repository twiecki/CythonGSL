import unittest, combinations, os

class CombinationTest(unittest.TestCase):

    def test_gsl_combination_calloc(self):
        t = combinations.t_gsl_combination_calloc()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_combination_init_first(self):
        t = combinations.t_gsl_combination_init_first()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_combination_memcpy(self):
        t = combinations.t_gsl_combination_memcpy()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_combination_n(self):
        t = combinations.t_gsl_combination_n()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_combination_get(self):
        t = combinations.t_gsl_combination_get()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_combination_valid(self):
        t = combinations.t_gsl_combination_valid()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_combination_fprintf(self):
        t = combinations.t_gsl_combination_fprintf()
        os.unlink('test.dat')
        for x in t:
            self.assertAlmostEqual(x,0, 15)

if __name__ == '__main__':
    unittest.main()
