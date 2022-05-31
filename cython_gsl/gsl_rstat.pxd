cdef extern from "gsl/gsl_rstat.h":

    ctypedef struct gsl_rstat_workspace
    ctypedef struct gsl_rstat_quantile_workspace

    gsl_rstat_workspace * gsl_rstat_alloc (void) nogil
    gsl_rstat_quantile_workspace * gsl_rstat_quantile_alloc (double p) nogil

    void gsl_rstat_free (gsl_rstat_workspace * w) nogil
    void gsl_rstat_quantile_free (gsl_rstat_quantile_workspace * w) nogil

    int gsl_rstat_reset (gsl_rstat_workspace * w) nogil
    int gsl_rstat_quantile_reset (gsl_rstat_quantile_workspace * w) nogil

    int gsl_rstat_add (double x, gsl_rstat_workspace * w) nogil
    int gsl_rstat_quantile_add (double x, gsl_rstat_quantile_workspace * w) nogil
    
    size_t gsl_rstat_n (gsl_rstat_workspace * w) nogil

    double gsl_rstat_min (gsl_rstat_workspace * w) nogil
    double gsl_rstat_max (gsl_rstat_workspace * w) nogil
    double gsl_rstat_mean (gsl_rstat_workspace * w) nogil
    double gsl_rstat_variance (gsl_rstat_workspace * w) nogil
    double gsl_rstat_sd (gsl_rstat_workspace * w) nogil
    double gsl_rstat_sd_mean (gsl_rstat_workspace * w) nogil
    double gsl_rstat_rms (gsl_rstat_workspace * w) nogil
    double gsl_rstat_skew (gsl_rstat_workspace * w) nogil
    double gsl_rstat_kurtosis (gsl_rstat_workspace * w) nogil
    double gsl_rstat_median (gsl_rstat_workspace * w) nogil

    double gsl_rstat_quantile_get (gsl_rstat_quantile_workspace * w) nogil