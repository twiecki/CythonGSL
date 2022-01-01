from cython_gsl cimport *

cdef extern from "gsl/gsl_dht.h":
  ctypedef struct gsl_dht_struct:
    size_t size  # size of the sample arrays to be transformed
    double nu    # Bessel function order                     
    double xmax  # the upper limit to the x-sampling domain 
    double kmax  # the upper limit to the k-sampling domain
    double * j     # array of computed J_nu zeros, j_{nu,s} = j[s]
    double * Jjj   # transform numerator, J_nu(j_i j_m / j_N)
    double * J2    # transform denominator, J_{nu+1}^2(j_m)
  ctypedef gsl_dht_struct gsl_dht

  # Create a new transform object for a given size
  # sampling array on the domain [0, xmax].
  gsl_dht * gsl_dht_alloc(size_t size) nogil
  gsl_dht * gsl_dht_new(size_t size, double nu, double xmax) nogil

  # Recalculate a transform object for given values of nu, xmax.
  # You cannot change the size of the object since the internal
  # allocation is reused.
  int gsl_dht_init(gsl_dht * t, double nu, double xmax) nogil

  # The n'th computed x sample point for a given transform.
  # 0 <= n <= size-1
  double gsl_dht_x_sample(gsl_dht * t, int n) nogil


  # The n'th computed k sample point for a given transform.
  # 0 <= n <= size-1
  double gsl_dht_k_sample(gsl_dht * t, int n) nogil


  # Free a transform object.
  void gsl_dht_free(gsl_dht * t) nogil


  # Perform a transform on a sampled array.
  # f_in[0] ... f_in[size-1] and similarly for f_out[]
  int gsl_dht_apply(gsl_dht * t, double * f_in, double * f_out) nogil

