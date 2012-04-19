from cython_gsl cimport *

def main():
    cdef double data[5]
    data[0] = 17.2
    data[1] = 18.1
    data[2] = 16.5
    data[3] = 18.3
    data[4] = 12.6
    cdef double median, upperq, lowerq

    print "Original dataset:  %g, %g, %g, %g, %g\n" % \
           (data[0], data[1], data[2], data[3], data[4])

    #gsl_sort (data, 1, 5)
    data[0] = 12.6
    data[1] = 16.5
    data[2] = 17.2
    data[3] = 18.1
    data[4] = 18.3

    print "Sorted dataset: %g, %g, %g, %g, %g\n" % \
           (data[0], data[1], data[2], data[3], data[4])

    median = gsl_stats_median_from_sorted_data (data, 1, 5)

    upperq = gsl_stats_quantile_from_sorted_data (data, 1, 5, 0.75)
    lowerq = gsl_stats_quantile_from_sorted_data (data, 1, 5, 0.25)

    print "The median is %g\n" % median
    print "The upper quartile is %g\n" % upperq
    print "The lower quartile is %g\n" % lowerq
