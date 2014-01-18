#    CythonGSL provides a set of Cython declarations for the GNU Scientific Library (GSL).
#    Copyright (C) 2012 Thomas V. Wiecki
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

from libc.math cimport *
from libc.stdio cimport *

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
from cython_gsl.gsl_mode cimport *
from cython_gsl.gsl_math cimport *
from cython_gsl.gsl_complex cimport *
from cython_gsl.gsl_poly cimport *
from cython_gsl.gsl_sf_result cimport *
from cython_gsl.gsl_airy cimport *
from cython_gsl.gsl_bessel cimport *
from cython_gsl.gsl_clausen cimport *
from cython_gsl.gsl_coulomb cimport *
from cython_gsl.gsl_coupling cimport *
from cython_gsl.gsl_dawson cimport *
from cython_gsl.gsl_debye cimport *
from cython_gsl.gsl_dilog cimport *
from cython_gsl.gsl_elementary cimport *
from cython_gsl.gsl_ellint cimport *
from cython_gsl.gsl_elljac cimport *
from cython_gsl.gsl_erf cimport *
from cython_gsl.gsl_exp cimport *
from cython_gsl.gsl_expint cimport *
from cython_gsl.gsl_fermi_dirac cimport *
from cython_gsl.gsl_gamma cimport *
from cython_gsl.gsl_gegenbauer cimport *
from cython_gsl.gsl_hyperg cimport *
from cython_gsl.gsl_laguerre cimport *
from cython_gsl.gsl_lambert cimport *
from cython_gsl.gsl_legendre cimport *
from cython_gsl.gsl_log cimport *
from cython_gsl.gsl_pow_int cimport *
from cython_gsl.gsl_psi cimport *
from cython_gsl.gsl_synchrotron cimport *
from cython_gsl.gsl_transport cimport *
from cython_gsl.gsl_trig cimport *
from cython_gsl.gsl_zeta cimport *
from cython_gsl.gsl_block cimport *
from cython_gsl.gsl_vector cimport *
from cython_gsl.gsl_vector_complex cimport *
from cython_gsl.gsl_matrix cimport *
from cython_gsl.gsl_matrix_complex cimport *
from cython_gsl.gsl_permutation cimport *
from cython_gsl.gsl_combination cimport *
from cython_gsl.gsl_sort cimport *
from cython_gsl.gsl_blas cimport *
from cython_gsl.gsl_blas_types cimport *
from cython_gsl.gsl_linalg cimport *
from cython_gsl.gsl_eigen cimport *
from cython_gsl.gsl_fft cimport *
from cython_gsl.gsl_integration cimport *
from cython_gsl.gsl_rng cimport *
from cython_gsl.gsl_qrng cimport *
from cython_gsl.gsl_random cimport *
from cython_gsl.gsl_statistics cimport *
from cython_gsl.gsl_histogram cimport *
from cython_gsl.gsl_ntuple cimport *
from cython_gsl.gsl_monte cimport *
from cython_gsl.gsl_odeiv cimport *
from cython_gsl.gsl_odeiv2 cimport *
from cython_gsl.gsl_interp cimport *
from cython_gsl.gsl_diff cimport *
from cython_gsl.gsl_chebyshev cimport *
from cython_gsl.gsl_sum cimport *
from cython_gsl.gsl_roots cimport *
from cython_gsl.gsl_min cimport *
from cython_gsl.gsl_fit cimport *
from cython_gsl.gsl_multimin cimport *
from cython_gsl.gsl_multifit_nlin cimport *
