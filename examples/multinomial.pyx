cimport cython
from cython_gsl cimport *

import numpy as np
from numpy cimport *

cdef gsl_rng *r = gsl_rng_alloc(gsl_rng_mt19937)

def multinomial(ndarray[double, ndim=1] p, unsigned int N):
    cdef:
       size_t K = p.shape[0]
       ndarray[uint32_t, ndim=1] n = np.empty_like(p, dtype='uint32')
    
    # void gsl_ran_multinomial (const gsl_rng * r, size_t K, unsigned int N, const double p[], unsigned int n[])
    gsl_ran_multinomial(r, K, N, <double*> p.data, <unsigned int *> n.data)
    
    return n

print multinomial(np.array([.2, .2, .2, .2, .2], dtype='double'), 500)
