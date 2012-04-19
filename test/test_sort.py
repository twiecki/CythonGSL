import unittest, sort, os

class SortTest(unittest.TestCase):

    def test_gsl_sort_smallest(self):
        t = sort.t_gsl_sort_smallest()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_sort_smallest_index(self):
        t = sort.t_gsl_sort_smallest_index()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_sort_vector_smallest(self):
        t = sort.t_gsl_sort_vector_smallest()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_sort_vector_smallest_index(self):
        t = sort.t_gsl_sort_vector_smallest_index()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_sort(self):
        t = sort.t_gsl_sort()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_sort_index(self):
        t = sort.t_gsl_sort_index()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_sort_vector(self):
        t = sort.t_gsl_sort_vector()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_sort_vector_index(self):
        t = sort.t_gsl_sort_vector_index()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_heapsort(self):
        t = sort.t_gsl_heapsort()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

    def test_gsl_heapsort_index(self):
        t = sort.t_gsl_heapsort_index()
        for x in t:
            self.assertAlmostEqual(x,0, 15)

if __name__ == '__main__':
    unittest.main()
