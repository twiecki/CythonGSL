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
from cython_gsl.gsl_interp cimport *
from cython_gsl.gsl_diff cimport *
from cython_gsl.gsl_chebyshev cimport *
from cython_gsl.gsl_sum cimport *
from cython_gsl.gsl_roots cimport *
from cython_gsl.gsl_min cimport *
from cython_gsl.gsl_fit cimport *
