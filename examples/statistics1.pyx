from cython_gsl cimport *

def main():
    cdef double data[5]
    data[0] = 17.2
    data[1] = 18.1
    data[2] = 16.5
    data[3] = 18.3
    data[4] = 12.6
    cdef double mean, variance, largest, smallest

    mean     = gsl_stats_mean(data, 1, 5)
    variance = gsl_stats_variance(data, 1, 5)
    largest  = gsl_stats_max(data, 1, 5)
    smallest = gsl_stats_min(data, 1, 5)

    print "The dataset is %g, %g, %g, %g, %g\n" % \
      (data[0], data[1], data[2], data[3], data[4])

    print "The sample mean is %g\n" % mean
    print"The estimated variance is %g\n" % variance
    print"The largest value is %g\n" % largest
    print"The smallest value is %g\n" % smallest
