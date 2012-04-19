from cython_gsl cimport *

def main():
    x = [2, 2, 4, 0,0,0]
    r = gsl_sf_coupling_3j(x[0], x[1],x[2],x[3],x[4],x[5])
    print r
