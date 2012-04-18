include "gsl.pxi"

def main():
  x = 1.0e-12
  cdef int e
  print gsl_frexp(x, &e)
  print e

