from cython_gsl cimport *

cdef struct data:
    double x
    double y
    double z



cdef int sel_func (void *ntuple_data, void *params) nogil:
    cdef data * data1
    data1 = <data *> ntuple_data
    cdef double x, y, z, E2, scale
    scale = (<double *> params)[0]

    x = data1.x
    y = data1.y
    z = data1.z

    E2 = x * x + y * y + z * z

    return E2 > scale

cdef double val_func (void *ntuple_data, void *params) nogil:
    cdef data * data1
    data1 = <data *> ntuple_data
    cdef double x, y, z

    x = data1.x
    y = data1.y
    z = data1.z

    return x * x + y * y + z * z

def main( ):
    cdef data ntuple_row

    cdef gsl_ntuple *ntuple
    ntuple  = gsl_ntuple_open ("test_dat/ntuple.dat", &ntuple_row, sizeof (ntuple_row))
    cdef double lower
    lower = 1.5

    cdef gsl_ntuple_select_fn S
    cdef gsl_ntuple_value_fn V

    cdef gsl_histogram *h
    h = gsl_histogram_alloc (100)
    gsl_histogram_set_ranges_uniform(h, 0.0, 10.0)

    S.function = &sel_func
    S.params = &lower

    V.function = &val_func
    V.params = NULL

    gsl_ntuple_project (h, ntuple, &V, &S)
    gsl_histogram_fprintf (stdout, h, "%f", "%f")
    gsl_histogram_free (h)
    gsl_ntuple_close (ntuple)
