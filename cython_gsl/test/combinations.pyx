from cython_gsl cimport *

def t_gsl_combination_calloc():
    cdef gsl_combination * c
    cdef size_t i
    cdef size_t * n
    cdef int status1
    c = gsl_combination_calloc (4, 2)
    ary = []
    n = gsl_combination_data(c)
    ary.extend([n[0], n[1] - 1])
    gsl_combination_next (c)
    ary.extend([n[0], n[1] - 2])
    gsl_combination_next (c)
    ary.extend([n[0], n[1] - 3])
    gsl_combination_next (c)
    ary.extend([n[0] - 1, n[1] - 2])
    status = gsl_combination_next (c)
    # status == 0 GSL_SUCCESS
    ary.extend([status, n[0] - 1, n[1] - 3])
    gsl_combination_next (c)
    ary.extend([n[0] - 2, n[1] - 3])
    status = gsl_combination_next (c)
    ary.extend([(status != 0) - 1, n[0] - 2, n[1] - 3])
    ary.extend([n[0] - 2, n[1] - 3])
    gsl_combination_prev(c)
    ary.extend([n[0] - 1, n[1] - 3])
    gsl_combination_free (c)
    return ary

def t_gsl_combination_init_first():
    cdef gsl_combination * c
    cdef size_t * n
    c = gsl_combination_calloc (4, 2)
    ary = []
    gsl_combination_next (c)
    gsl_combination_init_first(c)
    n = gsl_combination_data(c)
    ary.extend([n[0], n[1] - 1])
    gsl_combination_init_last(c)
    ary.extend([n[0] - 2, n[1] - 3])
    gsl_combination_free (c)
    return ary


def t_gsl_combination_memcpy():
    cdef gsl_combination * c1, * c2
    cdef size_t * n
    c1 = gsl_combination_calloc (4, 2)
    c2 = gsl_combination_calloc (4, 2)
    ary = []
    gsl_combination_next(c1)
    gsl_combination_memcpy(c2, c1)
    n = gsl_combination_data(c2)
    ary.extend([n[0], n[1] - 2])
    gsl_combination_free (c1)
    gsl_combination_free (c2)
    return ary


def t_gsl_combination_n():
    cdef gsl_combination * c
    c = gsl_combination_calloc (4, 2)
    ary = []
    gsl_combination_next (c)
    ary.append(gsl_combination_n(c) - 4)
    ary.append(gsl_combination_k(c) - 2)
    gsl_combination_free (c)
    return ary


def t_gsl_combination_get():
    cdef gsl_combination * c
    c = gsl_combination_calloc (4, 2)
    ary = []
    gsl_combination_next (c)
    ary.append(gsl_combination_get(c, 0))
    ary.append(gsl_combination_get(c, 1) - 2)
    gsl_combination_free (c)
    return ary

def t_gsl_combination_valid():
    cdef gsl_combination * c
    c = gsl_combination_calloc (4, 2)
    ary = []
    gsl_combination_next (c)
    ary.append(gsl_combination_valid(c))
    gsl_combination_free (c)
    cdef gsl_combination * c2
    c2 = gsl_combination_alloc (4, 2)
    if 0:
        c2.data[0] = 2
        c2.data[1] = 0
    else:
        c2.data[0] = 0
        c2.data[1] = 2
    ary.append(gsl_combination_valid(c2))
    gsl_combination_free (c2)
    return ary

def t_gsl_combination_fprintf():
    cdef gsl_combination * c
    c = gsl_combination_calloc (4, 2)
    ary = []
    gsl_combination_next (c)
    cdef FILE * f
    f = fopen ("test.dat", "w")
    gsl_combination_fprintf(f, c, "%u ")
    fclose(f)
    gsl_combination_free (c)
    cdef gsl_combination * c2
    c2 = gsl_combination_calloc (4, 2)
    cdef FILE * f2
    f2 = fopen ("test.dat", "r")
    gsl_combination_fscanf(f2, c2)
    fclose(f2)
    ary.append(gsl_combination_get(c2,0))
    ary.append(gsl_combination_get(c2,1) - 2)
    gsl_combination_free (c2)
    return ary
