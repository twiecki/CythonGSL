'''
Gibbs sampler for function:

f(x,y) = x x^2 \exp(-xy^2 - y^2 + 2y - 4x)

using conditional distributions:

x|y \sim Gamma(3, y^2 +4)
y|x \sim Normal(\frac{1}{1+x}, \frac{1}{2(1+x)})

Original code written by Chris Fonnesbeck.
Modified by Thomas V. Wiecki.
'''
cimport cython
from cython_gsl cimport *

import numpy as np
from numpy cimport *

cdef gsl_rng *r = gsl_rng_alloc(gsl_rng_mt19937)

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
def gibbs(int N=20000, int thin=500):
    cdef: 
        double x=0
        double y=0
        Py_ssize_t i, j
        ndarray[float64_t, ndim=2] samples

    samples = np.empty((N,thin))

    for i in range(N):
        for j in range(thin):
            x = gsl_ran_gamma(r,3,1.0/(y*y+4))
            y = gsl_ran_gaussian(r,1.0/sqrt(x+1))
        samples[i,0] = x
        samples[i,1] = y
    return samples
