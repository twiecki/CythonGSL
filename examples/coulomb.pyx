from cython_gsl cimport *

def main():
    x = [2.2, 0.4]
    Z = x[0]; r = x[1]
    res = gsl_sf_hydrogenicR_1(Z, r)
    print res

'''
puts "Normalized Hydrogenic Bound States"
x = [2.2, 0.4]; Z = x[0]; r = x[1]
res = Coulomb::hydrogenicR_1(*x)
assert res =~ (2 * Z * sqrt(Z) * exp(-Z*r))
'''
