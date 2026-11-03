#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <Rmath.h>
#include <R.h>
#include <Rinternals.h>
#include <R_ext/Utils.h>
// #include "general.h"

// Replacing general.h

double sq(double x) {
  return x * x;
}

double* new_vector(unsigned int n) {
  double *v = (double*) R_Calloc(n, double);
  return v;
}

double** new_matrix(unsigned int n1, unsigned int n2) {
  unsigned int i;
  double **m = (double**) R_Calloc(n1, double*);
  m[0] = (double*) R_Calloc(n1 * n2, double);
  for(i = 1; i < n1; i++) {
    m[i] = m[0] + i * n2;
  }
  return m;
}

double** new_matrix_bones(double *v, unsigned int n1, unsigned int n2) {
  unsigned int i;
  double **m = (double**) R_Calloc(n1, double*);
  for(i = 0; i < n1; i++) {
    m[i] = &v[i * n2];
  }
  return m;
}

void delete_matrix(double** m) {
  R_Free(m[0]);
  R_Free(m);
}

  


double sq(double x);

/* Small dense products used by ALC. Pure C avoids dependence on a local
 * Fortran toolchain or vendor-specific BLAS integer/character ABI. */
static void linalg_dsymv(int n, double alpha, double **A, int lda,
                         double *X, int ldx, double beta, double *Y, int ldy)
{
  int i, j;
  for (i = 0; i < n; ++i) {
    double sum = 0.0;
    for (j = 0; j < n; ++j) sum += A[i][j] * X[j * ldx];
    Y[i * ldy] = alpha * sum + beta * Y[i * ldy];
  }
}

static double linalg_ddot(int n, double *X, int ldx, double *Y, int ldy)
{
  int i;
  double result = 0.0;
  for (i = 0; i < n; ++i) result += X[i * ldx] * Y[i * ldy];
  return result;
}

/*
 * covar:
 *
 * calculate the correlation (K) between X1 and X2 with
 * an isotropic power exponential correlation function
 * with range d
 */

void covar(const int col, double **X1, const int n1, double **X2,
           const int n2, double d, double **K)
{
  int i, j, k;
  
  /* calculate the covariance */
  for(i=0; i<n1; i++)
    for(j=0; j<n2; j++) {
      K[i][j] = 0.0;
      for(k=0; k<col; k++) K[i][j] += sq(X1[i][k] - X2[j][k]);
      K[i][j] = exp(0.0 - K[i][j]/d);
    }
}

/*
 * calc_g_mui_kxy:
 *
 * function for calculating the g vector, mui scalar, and
 * kxy vector for the IECI calculation; kx is length-n
 * utility space -- only implements isotropic case
 */

void calc_g_mui_kxy(const int col, double *x, double **X,
                    const int n, double **Ki, double **Xref,
                    const int m, double d, double g, double *gvec,
                    double *mui, double *kx, double *kxy)
{
  double mu_neg;
  int i;
  
  /* sanity check */
  if(m == 0) assert(!kxy && !Xref);
  
  covar(col, &x, 1, X, n, d, &kx);
  if(m > 0) covar(col, &x, 1, Xref, m, d, &kxy);
  
  linalg_dsymv(n,1.0,Ki,n,kx,1,0.0,gvec,1);
  
  *mui = 1.0 + g - linalg_ddot(n, kx, 1, gvec, 1);
  
  mu_neg = 0.0 - 1.0/(*mui);
  for(i=0; i<n; i++) gvec[i] *= mu_neg;
}

/*
 * calc_ktKikx:
 *
 * function for calculating the ktKikx vector used in the
 * IECI calculation -- writes over the KtKik input
 */

void calc_ktKikx(double *ktKik, const int m, double **k, const int n,
                 double *g, const double mui, double *kxy, double **Gmui,
                 double *ktGmui, double *ktKikx)
{
  int i;
  
  /* loop over all of the m candidates */
  for(i=0; i<m; i++) {
    
    ktKikx[i] = sq(linalg_ddot(n, k[i], 1, g, 1))*mui;
    
    ktKikx[i] += 2.0*linalg_ddot(n, k[i], 1, g, 1)*kxy[i];
    
    ktKikx[i] += sq(kxy[i])/mui;
  }
}

/*
 * calc_alc:
 *
 * function that iterates over the m Xref locations, and the
 * stats calculated by previous calc_* function in order to
 * calculate the reduction in variance
 */

double calc_alc(const int m, double *ktKik, double *s2p, double *quad, int *q, const double phi,
                double *badj, const double tdf, double *w)
{
  int i;
  double alc;
  
  alc = 0.0;
  for(i=0; i<m; i++) {
    alc += pow(1 - (quad[i])- phi*ktKik[i], *q);
    // printf("quad[%d] = %f, ktKik[%d] = %f\n", i, quad[i], i, ktKik[i]);

  }
  
  return (alc/m);
}

/*
 * alcICMGP:
 *
 * return s2' component of the ALC calculation of the
 * expected reduction in variance calculation at locations
 * Xcand averaging over reference locations Xref:
 * ds2 = s2 - s2', where the s2s are at Xref and the
 * s2' incorporates Xcand, and everything is averaged
 * over Xref.
 */

 
void alcICMGP(int col, double **X, double **Ki, double d, double g,
           unsigned int n, double phi,
           unsigned int ncand, double **Xcand, unsigned int nref,
           double **Xref, double *quad, int *q, int trace, double *alc)
{
  int i;
  double **k;
  double *kx, *kxy, *gvec, *ktKikx;
  double mui, df;
  double s2p[2] = {0, 0};
  
  
  /* degrees of freedom */
  df = (double) n;
  
  /* allocate g, kxy, and ktKikx vectors */
  gvec = new_vector(n);
  kxy = new_vector(nref);
  kx = new_vector(n);
  ktKikx = new_vector(nref);
  
  k = new_matrix(nref, n);
  covar(col, Xref, nref, X, n, d, k);    //// CHANGED ncand to col
  
  /* calculate the ALC for each candidate */
  for(i=0; i<ncand; i++) {
    
    /* calculate the g vector, mui, and kxy */
    calc_g_mui_kxy(col, Xcand[i], X, n, Ki, Xref, nref, d,
                   g, gvec, &mui, kx, kxy);
    
    /* use g, mu, and kxy to calculate ktKik.x */
    calc_ktKikx(NULL, nref, k, n, gvec, mui, kxy, NULL, NULL, ktKikx);
    
    /* calculate the ALC */
    alc[i] = calc_alc(nref, ktKikx, s2p, quad, q, phi, NULL, df, NULL);
  }
  
  /* clean up */
  R_Free(ktKikx);
  R_Free(gvec);
  R_Free(kx);
  R_Free(kxy);
  delete_matrix(k);
}

/*
 * alcICMGP_R:
 *
 * R interface to C-side function that returns the
 * s2' component of the ALC calculation of the
 * expected reduction in variance calculation given
 * the stored GP parameterization at locations Xcand
 * averaging over reference locations Xref:
 * ds2 = s2 - s2', where the s2s are at Xref and the
 * s2' incorporates Xcand, and everything is averaged
 * over Xref.
 */

void alcICMGP_R(double *X_in, int *n_in, int *col_in, double *Ki_in, double *d_in,
             double *g_in, int *ncand_in, double *Xcand_in, int *nref_in,
             double *Xref_in, double *phi_in, double *quad, int *q, int *trace_in, double *alc_out)
{
  double **Xref, **Xcand, **X, **Ki;
  
  /* make matrix bones */
  Xref = new_matrix_bones(Xref_in, *nref_in, *col_in);
  Xcand = new_matrix_bones(Xcand_in, *ncand_in, *col_in);
  Ki = new_matrix_bones(Ki_in, *n_in, *n_in);
  X = new_matrix_bones(X_in, *n_in, *col_in);
  
  alcICMGP(*col_in, X, Ki, *d_in, *g_in, *n_in, *phi_in,
        *ncand_in, Xcand, *nref_in, Xref, quad, q, *trace_in, alc_out);
  
  /* clean up */
  R_Free(Xref);
  R_Free(Xcand);
  R_Free(Ki);
  R_Free(X);
}

