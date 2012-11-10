cdef extern from "gsl/gsl_odeiv2.h":
  
  ctypedef struct gsl_odeiv2_system:
    int (* function) (double t,  double y[], double dydt[], void * params) nogil
    int (* jacobian) (double t,  double y[], double * dfdy, double dfdt[], void * params) nogil
    size_t dimension
    void * params
  
  # no:
  #define GSL_ODEIV_FN_EVAL(S,t,y,f) (*((S)->function))(t,y,f,(S) nogil->params) nogil
  #define GSL_ODEIV_JA_EVAL(S,t,y,dfdy,dfdt)(*((S)->jacobian))(t,y,dfdy,dfdt,(S)->params) nogil
  
  
  
  
  ctypedef struct gsl_odeiv2_step
  ctypedef struct gsl_odeiv2_control
  ctypedef struct gsl_odeiv2_evolve
  ctypedef struct gsl_odeiv2_driver

  ctypedef struct gsl_odeiv2_step_type
  
  gsl_odeiv2_step_type *gsl_odeiv2_step_rk2
  gsl_odeiv2_step_type *gsl_odeiv2_step_rk4
  gsl_odeiv2_step_type *gsl_odeiv2_step_rkf45
  gsl_odeiv2_step_type *gsl_odeiv2_step_rkck
  gsl_odeiv2_step_type *gsl_odeiv2_step_rk8pd
  gsl_odeiv2_step_type *gsl_odeiv2_step_rk2imp
  gsl_odeiv2_step_type *gsl_odeiv2_step_rk4imp
  gsl_odeiv2_step_type *gsl_odeiv2_step_bsimp
  gsl_odeiv2_step_type *gsl_odeiv2_step_rk1imp
  gsl_odeiv2_step_type *gsl_odeiv2_step_msadams
  gsl_odeiv2_step_type *gsl_odeiv2_step_msbdf
  
  
  gsl_odeiv2_step * gsl_odeiv2_step_alloc( gsl_odeiv2_step_type * T, size_t dim) nogil
  int  gsl_odeiv2_step_reset(gsl_odeiv2_step * s) nogil
  void gsl_odeiv2_step_free(gsl_odeiv2_step * s) nogil
  
  char * gsl_odeiv2_step_name( gsl_odeiv2_step *) nogil
  unsigned int gsl_odeiv2_step_order( gsl_odeiv2_step * s) nogil
  
  int  gsl_odeiv2_step_apply(gsl_odeiv2_step *, double t, double h, double y[], double yerr[],  double dydt_in[], double dydt_out[],  gsl_odeiv2_system * dydt) nogil

  int  gsl_odeiv2_step_set_driver(gsl_odeiv2_step *, gsl_odeiv2_driver *) nogil
  
  
  #ctypedef struct gsl_odeiv2_control

  ctypedef struct gsl_odeiv2_control_type
  
  cdef enum:
    GSL_ODEIV_HADJ_DEC = -1
    GSL_ODEIV_HADJ_NIL = 0  
    GSL_ODEIV_HADJ_INC = 1  
  
  gsl_odeiv2_control * gsl_odeiv2_control_alloc( gsl_odeiv2_control_type * T) nogil
  int gsl_odeiv2_control_init(gsl_odeiv2_control * c, double eps_abs, double eps_rel, double a_y, double a_dydt) nogil
  void gsl_odeiv2_control_free(gsl_odeiv2_control * c) nogil
  int gsl_odeiv2_control_hadjust (gsl_odeiv2_control * c, gsl_odeiv2_step * s,  double y0[],  double yerr[],  double dydt[], double * h) nogil
  char * gsl_odeiv2_control_name(gsl_odeiv2_control * c) nogil
  int gsl_odeiv2_control_errlevel(gsl_odeiv2_control * c, double y,
                                  double dydt, double h,
                                  size_t ind, double *errlev) nogil
  int gsl_odeiv2_control_set_driver(gsl_odeiv2_control * c,
                                    gsl_odeiv2_driver *d) nogil
  
  
  gsl_odeiv2_control * gsl_odeiv2_control_standard_new(double eps_abs, double eps_rel, double a_y, double a_dydt) nogil
  gsl_odeiv2_control * gsl_odeiv2_control_y_new(double eps_abs, double eps_rel) nogil
  gsl_odeiv2_control * gsl_odeiv2_control_yp_new(double eps_abs, double eps_rel) nogil
  
  gsl_odeiv2_control * gsl_odeiv2_control_scaled_new(double eps_abs, double eps_rel, double a_y, double a_dydt,  double scale_abs[], size_t dim) nogil
  
  #ctypedef struct gsl_odeiv2_evolve
  
  gsl_odeiv2_evolve * gsl_odeiv2_evolve_alloc(size_t dim) nogil
  int gsl_odeiv2_evolve_apply(gsl_odeiv2_evolve *, gsl_odeiv2_control * con, gsl_odeiv2_step * step,  gsl_odeiv2_system * dydt, double * t, double t1, double * h, double y[]) nogil
  int gsl_odeiv2_evolve_apply_fixed_step(gsl_odeiv2_evolve * e,
    gsl_odeiv2_control * con, gsl_odeiv2_step * step,
    gsl_odeiv2_system * dydt, double *t, double h0, double y[]) nogil
  int gsl_odeiv2_evolve_reset(gsl_odeiv2_evolve *) nogil
  void gsl_odeiv2_evolve_free(gsl_odeiv2_evolve *) nogil
  int gsl_odeiv2_evolve_set_driver(gsl_odeiv2_evolve *e,
    gsl_odeiv2_driver *d) nogil

  #ctypedef struct gsl_odeiv2_driver

  gsl_odeiv2_driver *gsl_odeiv2_driver_alloc_y_new(
    gsl_odeiv2_system *sys, gsl_odeiv2_step_type *T,
    double hstart, double epsabs, double epsrel) nogil
  gsl_odeiv2_driver *gsl_odeiv2_driver_alloc_yp_new(
    gsl_odeiv2_system *sys, gsl_odeiv2_step_type *T,
    double hstart, double epsabs, double epsrel) nogil
  gsl_odeiv2_driver *gsl_odeiv2_driver_alloc_scaled_new(
    gsl_odeiv2_system *sys,
    gsl_odeiv2_step_type *T,
    double hstart, double epsabs, double epsrel,
    double a_y, double a_dydt, double scale_abs[]) nogil
  gsl_odeiv2_driver *gsl_odeiv2_driver_alloc_standard_new(
    gsl_odeiv2_system *sys,
    gsl_odeiv2_step_type *T,
    double hstart, double epsabs, double epsrel,
    double a_y, double a_dydt) nogil
  int gsl_odeiv2_driver_set_hmin(gsl_odeiv2_driver *d, double hmin) nogil
  int gsl_odeiv2_driver_set_hmax(gsl_odeiv2_driver *d, double hmax) nogil
  int gsl_odeiv2_driver_set_nmax(gsl_odeiv2_driver *d,
    unsigned long int nmax) nogil
  int gsl_odeiv2_driver_apply(gsl_odeiv2_driver *d,
    double *t, double t1, double y[]) nogil
  int gsl_odeiv2_driver_apply_fixed_step(gsl_odeiv2_driver *d,
    double *t, double h, unsigned long int n, double y[]) nogil
  int gsl_odeiv2_driver_reset(gsl_odeiv2_driver *d) nogil
  int gsl_odeiv2_driver_free(gsl_odeiv2_driver *d) nogil

