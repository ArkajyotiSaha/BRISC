#include <string>
#include <R.h>
#include <Rmath.h>
#include <Rinternals.h>
#include <R_ext/Linpack.h>
#include <R_ext/Lapack.h>
#include <R_ext/BLAS.h>
#include "util.h"

#ifdef _OPENMP
#include <omp.h>
#endif


extern "C" {

  SEXP BRISC_predictioncpp(SEXP X_r, SEXP y_r, SEXP coords_r, SEXP n_r, SEXP p_r, SEXP m_r, SEXP X0_r, SEXP coords0_r, SEXP q_r, SEXP nnIndx0_r,
		    SEXP betaSamples_r, SEXP thetaSamples_r, SEXP covModel_r, SEXP nThreads_r, SEXP verbose_r){

    int i, k, l, info, nProtect=0;
    const int inc = 1;
    const double one = 1.0;
    const double zero = 0.0;
    char const *lower = "L";


    //get args
    double *X = REAL(X_r);
    double *y = REAL(y_r);
    double *coords = REAL(coords_r);
    int n = INTEGER(n_r)[0];
    int p = INTEGER(p_r)[0];
    int m = INTEGER(m_r)[0];
    int mm = m*m;

    double *X0 = REAL(X0_r);
    double *coords0 = REAL(coords0_r);
    int q = INTEGER(q_r)[0];

    int *nnIndx0 = INTEGER(nnIndx0_r);
    double *beta = REAL(betaSamples_r);
    double *theta = REAL(thetaSamples_r);
    int covModel = INTEGER(covModel_r)[0];
    std::string corName = getCorName(covModel);
    int nThreads = INTEGER(nThreads_r)[0];
    int verbose = INTEGER(verbose_r)[0];

#ifdef _OPENMP
    omp_set_num_threads(nThreads);
#else
    if(nThreads > 1){
      warning("n.omp.threads > %i, but source not compiled with OpenMP support.", nThreads);
      nThreads = 1;
    }
#endif

    if(verbose){
      Rprintf("----------------------------------------\n");
      Rprintf("\tPrediction description\n");
      Rprintf("----------------------------------------\n");
      Rprintf("BRISC model fit with %i observations.\n\n", n);
      Rprintf("Number of covariates %i (including intercept if specified).\n\n", p);
      Rprintf("Using the %s spatial correlation model.\n\n", corName.c_str());
      Rprintf("Using %i nearest neighbors.\n\n", m);
      Rprintf("Predicting at %i locations.\n\n", q);
#ifdef _OPENMP
      Rprintf("\nSource compiled with OpenMP support and model fit using %i threads.\n", nThreads);
#else
      Rprintf("\n\nSource not compiled with OpenMP support.\n");
#endif
    }

    //parameters
    int nTheta, sigmaSqIndx, tauSqIndx, phiIndx, nuIndx;

    if(corName != "matern"){
      nTheta = 3;//sigma^2, tau^2, phi
      sigmaSqIndx = 0; tauSqIndx = 1; phiIndx = 2;
    }else{
      nTheta = 4;//sigma^2, tau^2, phi, nu
      sigmaSqIndx = 0; tauSqIndx = 1; phiIndx = 2; nuIndx = 3;
    }

    //get max nu
    double nuMax = 0;
    int nb = 0;

    if(corName == "matern"){
	if(theta[nuIndx] > nb){
	  nb = theta[nuIndx];
	}

      nb = 1+static_cast<int>(floor(nuMax));
    }

    double *bk = (double *) R_alloc(nThreads*nb, sizeof(double));

    double *C = (double *) R_alloc(nThreads*mm, sizeof(double)); zeros(C, nThreads*mm);
    double *c = (double *) R_alloc(nThreads*m, sizeof(double)); zeros(c, nThreads*m);
    double *tmp_m  = (double *) R_alloc(nThreads*m, sizeof(double));
    double phi = 0, nu = 0, sigmaSq = 0, tauSq = 0, d;
    int threadID = 0;

    SEXP y0_r;
    SEXP vary0_r;
    PROTECT(vary0_r = allocMatrix(REALSXP, q, 1)); nProtect++;
    double *vary0 = REAL(vary0_r);
    PROTECT(y0_r = allocMatrix(REALSXP, q, 1)); nProtect++;
    double *y0 = REAL(y0_r);

    if(verbose){
      Rprintf("-------------------------------------------------\n");
      Rprintf("\t\tPredicting\n");
      Rprintf("-------------------------------------------------\n");
      #ifdef Win32
        R_FlushConsole();
      #endif
    }

    double *z = (double *) R_alloc(q, sizeof(double));
    GetRNGstate();
    for(i = 0; i < q; i++){
      z[i] = rnorm(0.0,1.0);
    }
    PutRNGstate();




#ifdef _OPENMP
#pragma omp parallel for private(threadID, phi, nu, sigmaSq, tauSq, k, l, d, info)
#endif
    for(i = 0; i < q; i++){
#ifdef _OPENMP
	threadID = omp_get_thread_num();
#endif
	phi = theta[phiIndx];
	if(corName == "matern"){
	  nu = theta[nuIndx];
	}
	sigmaSq = theta[sigmaSqIndx];
	tauSq = theta[tauSqIndx];

	for(k = 0; k < m; k++){
	  d = dist2(coords[nnIndx0[i+q*k]], coords[n+nnIndx0[i+q*k]], coords0[i], coords0[q+i]);
	  c[threadID*m+k] = sigmaSq*spCor(d, phi, nu, covModel, &bk[threadID*nb]);
	  for(l = 0; l < m; l++){
	    d = dist2(coords[nnIndx0[i+q*k]], coords[n+nnIndx0[i+q*k]], coords[nnIndx0[i+q*l]], coords[n+nnIndx0[i+q*l]]);
	    C[threadID*mm+l*m+k] = sigmaSq*spCor(d, phi, nu, covModel, &bk[threadID*nb]);
	    if(k == l){
	      C[threadID*mm+l*m+k] += tauSq;
	    }
	  }
	}

	F77_NAME(dpotrf)(lower, &m, &C[threadID*mm], &m, &info); if(info != 0){error("c++ error: dpotrf failed\n");}
	F77_NAME(dpotri)(lower, &m, &C[threadID*mm], &m, &info); if(info != 0){error("c++ error: dpotri failed\n");}

	F77_NAME(dsymv)(lower, &m, &one, &C[threadID*mm], &m, &c[threadID*m], &inc, &zero, &tmp_m[threadID*m], &inc);

	d = 0;
	for(k = 0; k < m; k++){
	  d += tmp_m[threadID*m+k]*(y[nnIndx0[i+q*k]] - F77_NAME(ddot)(&p, &X[nnIndx0[i+q*k]], &n, &beta[0], &inc));
	}

	y0[i] = F77_NAME(ddot)(&p, &X0[i], &q, &beta[0], &inc) + d;
    vary0[i] = std::max(0.0, sqrt(sigmaSq + tauSq - F77_NAME(ddot)(&m, &tmp_m[threadID*m], &inc, &c[threadID*m], &inc)));
    }


    //make return object
    SEXP result_r, resultName_r;
    int nResultListObjs = 1 + 1;

    PROTECT(result_r = allocVector(VECSXP, nResultListObjs)); nProtect++;
    PROTECT(resultName_r = allocVector(VECSXP, nResultListObjs)); nProtect++;

    SET_VECTOR_ELT(result_r, 0, y0_r);
    SET_VECTOR_ELT(resultName_r, 0, mkChar("p.y.0"));

    SET_VECTOR_ELT(result_r, 1, vary0_r);
    SET_VECTOR_ELT(resultName_r, 1, mkChar("var.y.0"));

    namesgets(result_r, resultName_r);

    //unprotect
    UNPROTECT(nProtect);

    return(result_r);

  }
}
