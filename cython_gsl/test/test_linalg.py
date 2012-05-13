import unittest, linalg, os

class LinalgTest(unittest.TestCase):

    def test_gsl_linalg_LU_decomp(self):
        t = linalg.t_gsl_linalg_LU_decomp()
        for x in t:
            self.assertAlmostEqual(x,0, 14)


    def test_gsl_linalg_LU_solve(self):
        t = linalg.t_gsl_linalg_LU_solve()
        for x in t:
            self.assertAlmostEqual(x,0, 14)

    def test_gsl_linalg_LU_svx(self):
        t = linalg.t_gsl_linalg_LU_svx()
        for x in t:
            self.assertAlmostEqual(x,0, 14)

    @unittest.expectedFailure
    def test_gsl_linalg_LU_refine(self):
        t = linalg.t_gsl_linalg_LU_refine()
        for x in t:
            self.assertAlmostEqual(x,0, 14)

    def test_gsl_linalg_LU_invert(self):
        t = linalg.t_gsl_linalg_LU_invert()
        for x in t:
            self.assertAlmostEqual(x,0, 14)

    def test_gsl_linalg_LU_det(self):
        t = linalg.t_gsl_linalg_LU_det()
        for x in t:
            self.assertAlmostEqual(x,0, 14)

    def test_gsl_linalg_householder_transform(self):
        t = linalg.t_gsl_linalg_householder_transform()
        for x in t:
            self.assertAlmostEqual(x,0, 14)

    def test_gsl_linalg_householder_hm(self):
        t = linalg.t_gsl_linalg_householder_hm()
        for x in t:
            self.assertAlmostEqual(x,0, 14)

    def test_gsl_linalg_householder_mh(self):
        t = linalg.t_gsl_linalg_householder_mh()
        for x in t:
            self.assertAlmostEqual(x,0, 14)

    def test_gsl_linalg_QR_decomp(self):
        t = linalg.t_gsl_linalg_QR_decomp()
        for x in t:
            self.assertAlmostEqual(x,0, 14)

    def test_gsl_linalg_QR_solve(self):
        t = linalg.t_gsl_linalg_QR_solve()
        for x in t:
            self.assertAlmostEqual(x,0, 14)

    def test_gsl_linalg_QR_svx(self):
        t = linalg.t_gsl_linalg_QR_svx()
        for x in t:
            self.assertAlmostEqual(x,0, 14)

    def test_gsl_linalg_QR_lssolve(self):
        t = linalg.t_gsl_linalg_QR_lssolve()
        for x in t:
            self.assertAlmostEqual(x,0, 14)

    def test_gsl_linalg_QR_Qvec(self):
        t = linalg.t_gsl_linalg_QR_Qvec()
        for x in t:
            self.assertAlmostEqual(x,0, 14)

    def test_gsl_linalg_R_solve(self):
        t = linalg.t_gsl_linalg_R_solve()
        for x in t:
            self.assertAlmostEqual(x,0, 14)

    def test_gsl_linalg_QR_QRsolve(self):
        t = linalg.t_gsl_linalg_QR_QRsolve()
        for x in t:
            self.assertAlmostEqual(x,0, 14)

    def test_gsl_linalg_SV_decomp(self):
        t = linalg.t_gsl_linalg_SV_decomp()
        for x in t:
            self.assertAlmostEqual(x,0, 14)

    def test_gsl_linalg_SV_decomp_mod(self):
        t = linalg.t_gsl_linalg_SV_decomp_mod()
        for x in t:
            self.assertAlmostEqual(x,0, 14)

    def test_gsl_linalg_SV_decomp_jacobi(self):
        t = linalg.t_gsl_linalg_SV_decomp_jacobi()
        for x in t:
            self.assertAlmostEqual(x,0, 14)

    def test_gsl_linalg_cholesky_decomp(self):
        t = linalg.t_gsl_linalg_cholesky_decomp()
        for x in t:
            self.assertAlmostEqual(x,0, 14)

    def test_gsl_linalg_cholesky_solve(self):
        t = linalg.t_gsl_linalg_cholesky_solve()
        for x in t:
            self.assertAlmostEqual(x,0, 14)

    def test_gsl_linalg_HH_solve(self):
        t = linalg.t_gsl_linalg_HH_solve()
        for x in t:
            self.assertAlmostEqual(x,0, 14)

    def test_gsl_linalg_solve_symm_tridiag(self):
        t = linalg.t_gsl_linalg_solve_symm_tridiag()
        for x in t:
            self.assertAlmostEqual(x,0, 14)

    def test_gsl_linalg_solve_symm_cyc_tridiag(self):
        t = linalg.t_gsl_linalg_solve_symm_cyc_tridiag()
        for x in t:
            self.assertAlmostEqual(x,0, 14)

    def test_gsl_linalg_symmtd_decomp(self):
        t = linalg.t_gsl_linalg_symmtd_decomp()
        for x in t:
            self.assertAlmostEqual(x,0, 14)

    def test_gsl_linalg_bidiag_decomp(self):
        t = linalg.t_gsl_linalg_bidiag_decomp()
        for x in t:
            self.assertAlmostEqual(x,0, 14)

if __name__ == '__main__':
    unittest.main()
