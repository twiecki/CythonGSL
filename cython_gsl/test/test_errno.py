import unittest, gslerrno

class ErrnoTest(unittest.TestCase):

    def test_gsl_matrix_alloc_fail(self):
        t = gslerrno.t_gsl_matrix_alloc_fail()

if __name__ == '__main__':
    unittest.main()
