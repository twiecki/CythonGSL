from cython_gsl import gsl

cpdef Normal(double x, double sigma):
    return gsl.gsl_ran_gaussian_pdf(x, sigma)

