from cython_gsl cimport *

def main():
    cdef double sn, cn, dn
    gsl_sf_elljac_e(10.2, 0.2, &sn, &cn, &dn)
    print sn, cn, dn
