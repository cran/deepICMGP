#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>

/* Declare your C functions */
void alcICMGP_R(double *X_in, int *n_in, int *col_in, double *Ki_in, double *d_in,
             double *g_in, int *ncand_in, double *Xcand_in, int *nref_in,
             double *Xref_in, double *phi_in, double *quad, int *q, int *trace_in, double *alc_out);

/* Register routines */
static const R_CMethodDef CEntries[] = {
  {"alcICMGP_R", (DL_FUNC) &alcICMGP_R, 15},
  {NULL, NULL, 0}
};

void R_init_deepICMGP(DllInfo *dll) {
  R_registerRoutines(dll, CEntries, NULL, NULL, NULL);
  R_useDynamicSymbols(dll, FALSE);
}
