cimport cython_gsl.math
cimport cython_gsl.stdio
cdef enum:
  GSL_SUCCESS = 0
  GSL_FAILURE  = -1
  GSL_CONTINUE = -2  # iteration has not converged
  GSL_EDOM     = 1   # input domain error, e.g sqrt(-1)
  GSL_ERANGE   = 2   # output range error, e.g. exp(1e100)
  GSL_EFAULT   = 3   # invalid pointer
  GSL_EINVAL   = 4   # invalid argument supplied by user
  GSL_EFAILED  = 5   # generic failure
  GSL_EFACTOR  = 6   # factorization failed
  GSL_ESANITY  = 7   # sanity check failed - shouldn't happen
  GSL_ENOMEM   = 8   # malloc failed
  GSL_EBADFUNC = 9   # problem with user-supplied function
  GSL_ERUNAWAY = 10  # iterative process is out of control
  GSL_EMAXITER = 11  # exceeded max number of iterations
  GSL_EZERODIV = 12  # tried to divide by zero
  GSL_EBADTOL  = 13  # user specified an invalid tolerance
  GSL_ETOL     = 14  # failed to reach the specified tolerance
  GSL_EUNDRFLW = 15  # underflow
  GSL_EOVRFLW  = 16  # overflow
  GSL_ELOSS    = 17  # loss of accuracy
  GSL_EROUND   = 18  # failed because of roundoff error
  GSL_EBADLEN  = 19  # matrix, vector lengths are not conformant
  GSL_ENOTSQR  = 20  # matrix not square
  GSL_ESING    = 21  # apparent singularity detected
  GSL_EDIVERGE = 22  # integral or series is divergent
  GSL_EUNSUP   = 23  # requested feature is not supported by the hardware
  GSL_EUNIMPL  = 24  # requested feature not (yet) implemented
  GSL_ECACHE   = 25  # cache limit exceeded
  GSL_ETABLE   = 26  # table limit exceeded
  GSL_ENOPROG  = 27  # iteration is not making progress towards solution
  GSL_ENOPROGJ = 28  # jacobian evaluations are not improving the solution
  GSL_ETOLF    = 29  # cannot reach the specified tolerance in F
  GSL_ETOLX    = 30  # cannot reach the specified tolerance in X
  GSL_ETOLG    = 31  # cannot reach the specified tolerance in gradient
  GSL_EOF      = 32  # end of file

ctypedef int size_t
cimport cython_gsl.gsl_mode
cimport cython_gsl.gsl_math
cimport cython_gsl.gsl_complex
cimport cython_gsl.gsl_poly
cimport cython_gsl.gsl_sf_result
cimport cython_gsl.gsl_airy
cimport cython_gsl.gsl_bessel
cimport cython_gsl.gsl_clausen
cimport cython_gsl.gsl_coulomb
cimport cython_gsl.gsl_coupling
cimport cython_gsl.gsl_dawson
cimport cython_gsl.gsl_debye
cimport cython_gsl.gsl_dilog
cimport cython_gsl.gsl_elementary
cimport cython_gsl.gsl_ellint
cimport cython_gsl.gsl_elljac
cimport cython_gsl.gsl_erf
cimport cython_gsl.gsl_exp
cimport cython_gsl.gsl_expint
cimport cython_gsl.gsl_fermi_dirac
cimport cython_gsl.gsl_gamma
cimport cython_gsl.gsl_gegenbauer
cimport cython_gsl.gsl_hyperg
cimport cython_gsl.gsl_laguerre
cimport cython_gsl.gsl_lambert
cimport cython_gsl.gsl_legendre
cimport cython_gsl.gsl_log
cimport cython_gsl.gsl_pow_int
cimport cython_gsl.gsl_psi
cimport cython_gsl.gsl_synchrotron
cimport cython_gsl.gsl_transport
cimport cython_gsl.gsl_trig
cimport cython_gsl.gsl_zeta

cimport cython_gsl.gsl_block
cimport cython_gsl.gsl_vector
cimport cython_gsl.gsl_vector_complex
cimport cython_gsl.gsl_matrix
cimport cython_gsl.gsl_matrix_complex

cimport cython_gsl.gsl_permutation
cimport cython_gsl.gsl_combination
cimport cython_gsl.gsl_sort

cimport cython_gsl.gsl_blas
cimport cython_gsl.gsl_linalg
cimport cython_gsl.gsl_eigen
cimport cython_gsl.gsl_fft
cimport cython_gsl.gsl_integration
cimport cython_gsl.gsl_rng
cimport cython_gsl.gsl_qrng
cimport cython_gsl.gsl_random
cimport cython_gsl.gsl_statistics
cimport cython_gsl.gsl_histogram
cimport cython_gsl.gsl_ntuple
cimport cython_gsl.gsl_monte
cimport cython_gsl.gsl_odeiv
cimport cython_gsl.gsl_odeiv2
cimport cython_gsl.gsl_interp
cimport cython_gsl.gsl_diff
cimport cython_gsl.gsl_chebyshev
cimport cython_gsl.gsl_sum
cimport cython_gsl.gsl_roots
cimport cython_gsl.gsl_min
cimport cython_gsl.gsl_fit
