from cython_gsl cimport *

cdef extern from "gsl/gsl_heapsort.h":

  ctypedef int (*gsl_comparison_fn_t) ( void * ,  void * ) nogil

  void gsl_heapsort (void * array, size_t count, size_t size, gsl_comparison_fn_t compare) nogil
  int gsl_heapsort_index (size_t * p,  void * array, size_t count, size_t size, gsl_comparison_fn_t compare) nogil


cdef extern from "gsl/gsl_sort_double.h":
  void gsl_sort (double * data,  size_t stride,  size_t n) nogil
  void gsl_sort_index (size_t * p,  double * data,  size_t stride,  size_t n) nogil

  int gsl_sort_smallest (double * dest,  size_t k,  double * src,  size_t stride,  size_t n) nogil
  int gsl_sort_smallest_index (size_t * p,  size_t k,  double * src,  size_t stride,  size_t n) nogil

  int gsl_sort_largest (double * dest,  size_t k,  double * src,  size_t stride,  size_t n) nogil
  int gsl_sort_largest_index (size_t * p,  size_t k,  double * src,  size_t stride,  size_t n) nogil


cdef extern from "gsl/gsl_sort_vector_double.h":

  void gsl_sort_vector (gsl_vector * v) nogil
  int gsl_sort_vector_index (gsl_permutation * p,  gsl_vector * v) nogil

  int gsl_sort_vector_smallest (double * dest,  size_t k,  gsl_vector * v) nogil
  int gsl_sort_vector_largest (double * dest,  size_t k,  gsl_vector * v) nogil

  int gsl_sort_vector_smallest_index (size_t * p,  size_t k,  gsl_vector * v) nogil
  int gsl_sort_vector_largest_index (size_t * p,  size_t k,  gsl_vector * v) nogil

