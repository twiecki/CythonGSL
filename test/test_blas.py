import unittest, blas, os

class BlasTest(unittest.TestCase):

    def test_gsl_blas_ddot(self):
        t = blas.t_gsl_blas_ddot()
        for x in t:
            self.assertAlmostEqual(x,0, 14)

    def test_gsl_blas_zdotu(self):
        t = blas.t_gsl_blas_zdotu()
        for x in t:
            self.assertAlmostEqual(x,0, 14)

    def test_gsl_blas_dnrm2(self):
        t = blas.t_gsl_blas_dnrm2()
        for x in t:
            self.assertAlmostEqual(x,0, 14)

    def test_gsl_blas_dasum(self):
        t = blas.t_gsl_blas_dasum()
        for x in t:
            self.assertAlmostEqual(x,0, 14)

    def test_gsl_blas_dznrm2(self):
        t = blas.t_gsl_blas_dznrm2()
        for x in t:
            self.assertAlmostEqual(x,0, 14)

    def test_gsl_blas_dzasum(self):
        t = blas.t_gsl_blas_dzasum()
        for x in t:
            self.assertAlmostEqual(x,0, 14)

    def test_gsl_blas_dswap(self):
        t = blas.t_gsl_blas_dswap()
        for x in t:
            self.assertAlmostEqual(x,0, 14)

    def test_gsl_blas_dcopy(self):
        t = blas.t_gsl_blas_dcopy()
        for x in t:
            self.assertAlmostEqual(x,0, 14)

    def test_gsl_blas_daxpy(self):
        t = blas.t_gsl_blas_daxpy()
        for x in t:
            self.assertAlmostEqual(x,0, 14)

    def test_gsl_blas_zswap(self):
        t = blas.t_gsl_blas_zswap()
        for x in t:
            self.assertAlmostEqual(x,0, 14)

    def test_gsl_blas_zcopy(self):
        t = blas.t_gsl_blas_zcopy()
        for x in t:
            self.assertAlmostEqual(x,0, 14)

    def test_gsl_blas_zaxpy(self):
        t = blas.t_gsl_blas_zaxpy()
        for x in t:
            self.assertAlmostEqual(x,0, 14)

    def test_gsl_blas_drotg(self):
        t = blas.t_gsl_blas_drotg()
        for x in t:
            self.assertAlmostEqual(x,0, 14)

    def test_gsl_blas_dscal(self):
        t = blas.t_gsl_blas_dscal()
        for x in t:
            self.assertAlmostEqual(x,0, 14)

    def test_gsl_blas_zscal(self):
        t = blas.t_gsl_blas_zscal()
        for x in t:
            self.assertAlmostEqual(x,0, 14)

    def test_gsl_blas_zdscal(self):
        t = blas.t_gsl_blas_zdscal()
        for x in t:
            self.assertAlmostEqual(x,0, 14)

    def test_gsl_blas_dgemv(self):
        t = blas.t_gsl_blas_dgemv()
        for x in t:
            self.assertAlmostEqual(x,0, 14)

    def test_gsl_blas_dtrmv(self):
        t = blas.t_gsl_blas_dtrmv()
        for x in t:
            self.assertAlmostEqual(x,0, 14)

    def test_gsl_blas_dtrsv(self):
        t = blas.t_gsl_blas_dtrsv()
        for x in t:
            self.assertAlmostEqual(x,0, 14)

    def test_gsl_blas_dsymv(self):
        t = blas.t_gsl_blas_dsymv()
        for x in t:
            self.assertAlmostEqual(x,0, 14)

    def test_gsl_blas_dger(self):
        t = blas.t_gsl_blas_dger()
        for x in t:
            self.assertAlmostEqual(x,0, 14)

    def test_gsl_blas_dsyr(self):
        t = blas.t_gsl_blas_dsyr()
        for x in t:
            self.assertAlmostEqual(x,0, 14)

    def test_gsl_blas_dsyr2(self):
        t = blas.t_gsl_blas_dsyr2()
        for x in t:
            self.assertAlmostEqual(x,0, 14)

    def test_gsl_blas_zgemv(self):
        t = blas.t_gsl_blas_zgemv()
        for x in t:
            self.assertAlmostEqual(x,0, 14)

    def test_gsl_blas_ztrmv(self):
        t = blas.t_gsl_blas_ztrmv()
        for x in t:
            self.assertAlmostEqual(x,0, 14)

    def test_gsl_blas_ztrsv(self):
        t = blas.t_gsl_blas_ztrsv()
        for x in t:
            self.assertAlmostEqual(x,0, 14)

    def test_gsl_blas_zhemv(self):
        t = blas.t_gsl_blas_zhemv()
        for x in t:
            self.assertAlmostEqual(x,0, 14)

    def test_gsl_blas_zgeru(self):
        t = blas.t_gsl_blas_zgeru()
        for x in t:
            self.assertAlmostEqual(x,0, 14)

    def test_gsl_blas_zgerc(self):
        t = blas.t_gsl_blas_zgerc()
        for x in t:
            self.assertAlmostEqual(x,0, 14)

    def test_gsl_blas_zher(self):
        t = blas.t_gsl_blas_zher()
        for x in t:
            self.assertAlmostEqual(x,0, 14)

    def test_gsl_blas_zher2(self):
        t = blas.t_gsl_blas_zher2()
        for x in t:
            self.assertAlmostEqual(x,0, 14)

    def test_gsl_blas_dgemm(self):
        t = blas.t_gsl_blas_dgemm()
        for x in t:
            self.assertAlmostEqual(x,0, 14)

    def test_gsl_blas_dsymm(self):
        t = blas.t_gsl_blas_dsymm()
        for x in t:
            self.assertAlmostEqual(x,0, 14)

    def test_gsl_blas_dtrmm(self):
        t = blas.t_gsl_blas_dtrmm()
        for x in t:
            self.assertAlmostEqual(x,0, 14)

    def test_gsl_blas_dtrsm(self):
        t = blas.t_gsl_blas_dtrsm()
        for x in t:
            self.assertAlmostEqual(x,0, 14)

    def test_gsl_blas_dsyrk(self):
        t = blas.t_gsl_blas_dsyrk()
        for x in t:
            self.assertAlmostEqual(x,0, 14)

    def test_gsl_blas_dsyr2k(self):
        t = blas.t_gsl_blas_dsyr2k()
        for x in t:
            self.assertAlmostEqual(x,0, 14)

    def test_gsl_blas_zgemm(self):
        t = blas.t_gsl_blas_zgemm()
        for x in t:
            self.assertAlmostEqual(x,0, 14)

    def test_gsl_blas_zsymm(self):
        t = blas.t_gsl_blas_zsymm()
        for x in t:
            self.assertAlmostEqual(x,0, 14)

    def test_gsl_blas_zsyrk(self):
        t = blas.t_gsl_blas_zsyrk()
        for x in t:
            self.assertAlmostEqual(x,0, 14)

    def test_gsl_blas_zsyr2k(self):
        t = blas.t_gsl_blas_zsyr2k()
        for x in t:
            self.assertAlmostEqual(x,0, 13)

    def test_gsl_blas_ztrmm(self):
        t = blas.t_gsl_blas_ztrmm()
        for x in t:
            self.assertAlmostEqual(x,0, 14)

    def test_gsl_blas_ztrsm(self):
        t = blas.t_gsl_blas_ztrsm()
        for x in t:
            self.assertAlmostEqual(x,0, 14)

    def test_gsl_blas_zhemm(self):
        t = blas.t_gsl_blas_zhemm()
        for x in t:
            self.assertAlmostEqual(x,0, 14)

    def test_gsl_blas_zherk(self):
        t = blas.t_gsl_blas_zherk()
        for x in t:
            self.assertAlmostEqual(x,0, 14)

if __name__ == '__main__':
    unittest.main()
