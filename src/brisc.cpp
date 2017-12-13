#include <string>
#include <R.h>
#include <Rmath.h>
#include <Rinternals.h>
#include <R_ext/Linpack.h>
#include <R_ext/Lapack.h>
#include <R_ext/BLAS.h>
#include <stdio.h>
#include <limits>
#include "lbfgs.h"
#include "util.h"

#ifdef _OPENMP
#include <omp.h>
#endif

void zeros(double *a, int n){
    for(int i = 0; i < n; i++)
        a[i] = 0.0;
}

void mvrnorm(double *des, double *mu, double *cholCov, int dim){
    
    int i;
    int inc = 1;
    double one = 1.0;
    
    for(i = 0; i < dim; i++){
        des[i] = rnorm(0, 1);
    }
    
    F77_NAME(dtrmv)("L", "N", "N", &dim, cholCov, &dim, des, &inc);
    F77_NAME(daxpy)(&dim, &one, mu, &inc, des, &inc);
}

double logit(double theta, double a, double b){
    return log((theta-a)/(b-theta));
}

double logitInv(double z, double a, double b){
    return b-(b-a)/(1+exp(z));
}

double dist2(double &a1, double &a2, double &b1, double &b2){
    return(sqrt(pow(a1-b1,2)+pow(a2-b2,2)));
}

void getNNIndx(int i, int m, int &iNNIndx, int &iNN){
    
    if(i == 0){
        iNNIndx = 0;//this should never be accessed
        iNN = 0;
        return;
    }else if(i < m){
        iNNIndx = static_cast<int>(static_cast<double>(i)/2*(i-1));
        iNN = i;
        return;
    }else{
        iNNIndx = static_cast<int>(static_cast<double>(m)/2*(m-1)+(i-m)*m);
        iNN = m;
        return;
    }
}

void mkNNIndx(int n, int m, double *coords, int *nnIndx, double *nnDist, int *nnIndxLU){
    
    int i, j, iNNIndx, iNN;
    double d;
    
    int nIndx = static_cast<int>(static_cast<double>(1+m)/2*m+(n-m-1)*m);
    
    for(i = 0; i < nIndx; i++){
        nnDist[i] = std::numeric_limits<double>::infinity();
    }
    
#ifdef _OPENMP
#pragma omp parallel for private(j, iNNIndx, iNN, d)
#endif
    for(i = 0; i < n; i++){
        getNNIndx(i, m, iNNIndx, iNN);
        nnIndxLU[i] = iNNIndx;
        nnIndxLU[n+i] = iNN;
        if(i != 0){
            for(j = 0; j < i; j++){
                d = dist2(coords[i], coords[n+i], coords[j], coords[n+j]);
                if(d < nnDist[iNNIndx+iNN-1]){
                    nnDist[iNNIndx+iNN-1] = d;
                    nnIndx[iNNIndx+iNN-1] = j;
                    rsort_with_index(&nnDist[iNNIndx], &nnIndx[iNNIndx], iNN);
                }
            }
        }
    }
    
}

void mkUIndx(int n, int m, int* nnIndx, int* uIndx, int* uIndxLU){
    
    int iNNIndx, iNN, i, j, k, l, h;
    
    for(i = 0, l = 0; i < n; i++){
        uIndxLU[i] = l;
        for(j = 0, h = 0; j < n; j++){
            getNNIndx(j, m, iNNIndx, iNN);
            for(k = 0; k < iNN; k++){
                if(nnIndx[iNNIndx+k] == i){
                    uIndx[l+h] = j;
                    h++;
                }
            }
        }
        l += h;
        uIndxLU[n+i] = h;
    }
}

std::string getCorName(int i){
    
    if(i == 0){
        return "exponential";
    }else if(i == 1){
        return "spherical";
    }else if(i == 2){
        return "matern";
    }else if(i == 3){
        return "gaussian";
    }else{
        error("c++ error: cov.model is not correctly specified");
    }
    
}

double spCor(double &D, double &phi, double &nu, int &covModel, double *bk){
    
    //0 exponential
    //1 spherical
    //2 matern
    //3 gaussian
    
    if(covModel == 0){//exponential
        
        return exp(-phi*D);
        
    }else if(covModel == 1){//spherical
        
        if(D > 0 && D <= 1.0/phi){
            return 1.0 - 1.5*phi*D + 0.5*pow(phi*D,3);
        }else if(D >= 1.0/phi){
            return 0.0;
        }else{
            return 1.0;
        }
    }else if(covModel == 2){//matern
        
        //(d*phi)^nu/(2^(nu-1)*gamma(nu))*pi/2*(besselI(d*phi,-nu)-besselI(d*phi, nu))/sin(nu*pi), or
        //(d*phi)^nu/(2^(nu-1)*gamma(nu))*besselK(x=d*phi, nu=nu)
        
        if(D*phi > 0.0){
            return pow(D*phi, nu)/(pow(2, nu-1)*gammafn(nu))*bessel_k_ex(D*phi, nu, 1.0, bk);//thread safe bessel
        }else{
            return 1.0;
        }
    }else if(covModel == 3){//gaussian
        
        return exp(-1.0*(pow(phi*D,2)));
        
    }else{
        error("c++ error: cov.model is not correctly specified");
    }
}

//which index of b equals a, where b is of length n
int which(int a, int *b, int n){
    int i;
    for(i = 0; i < n; i++){
        if(a == b[i]){
            return(i);
        }
    }
    
    error("c++ error: which failed");
    return -9999;
}

//Description: computes the quadratic term.
double Q(double *B, double *F, double *u, double *v, int n, int *nnIndx, int *nnIndxLU){
    
    double a, b, q = 0;
    int i, j;
    
#ifdef _OPENMP
#pragma omp parallel for private(a, b, j) reduction(+:q)
#endif
    for(i = 0; i < n; i++){
        a = 0;
        b = 0;
        for(j = 0; j < nnIndxLU[n+i]; j++){
            a += B[nnIndxLU[i]+j]*u[nnIndx[nnIndxLU[i]+j]];
            b += B[nnIndxLU[i]+j]*v[nnIndx[nnIndxLU[i]+j]];
        }
        q += (u[i] - a)*(v[i] - b)/F[i];
    }
    
    return(q);
}

//trees
Node *miniInsert(Node *Tree, double *coords, int index, int d,int n){
    
    int P = 2;
    
    if(Tree==NULL){
        return new Node(index);
    }
    
    if(coords[index]<=coords[Tree->index]&&d==0){
        Tree->left=miniInsert(Tree->left,coords,index,(d+1)%P,n);
    }
    
    if(coords[index]>coords[Tree->index]&&d==0){
        Tree->right=miniInsert(Tree->right,coords,index,(d+1)%P,n);
    }
    
    if(coords[index+n]<=coords[Tree->index+n]&&d==1){
        Tree->left=miniInsert(Tree->left,coords,index,(d+1)%P,n);
    }
    
    if(coords[index+n]>coords[Tree->index+n]&&d==1){
        Tree->right=miniInsert(Tree->right,coords,index,(d+1)%P,n);
    }
    
    return Tree;
}

void get_nn(Node *Tree, int index, int d, double *coords, int n, double *nnDist, int *nnIndx, int iNNIndx, int iNN, int check){
    
    int P = 2;
    
    if(Tree==NULL){
        return;
    }
    
    double disttemp= dist2(coords[index],coords[index+n],coords[Tree->index],coords[Tree->index+n]);
    
    if(index!=Tree->index && disttemp<nnDist[iNNIndx+iNN-1]){
        nnDist[iNNIndx+iNN-1]=disttemp;
        nnIndx[iNNIndx+iNN-1]=Tree->index;
        //fSort(&nnDist[iNNIndx], &nnIndx[iNNIndx], iNN);
        rsort_with_index(&nnDist[iNNIndx], &nnIndx[iNNIndx], iNN);
    }
    
    Node *temp1=Tree->left;
    Node *temp2=Tree->right;
    
    if(d==0){
        
        if(coords[index]>coords[Tree->index]){
            std::swap(temp1,temp2);
        }
        
        get_nn(temp1,index,(d+1)%P,coords,n, nnDist, nnIndx, iNNIndx, iNN, check);
        
        if(abs(coords[Tree->index]-coords[index])>nnDist[iNNIndx+iNN-1]){
            return;
        }
        
        get_nn(temp2,index,(d+1)%P,coords,n, nnDist, nnIndx, iNNIndx, iNN, check);
    }
    
    if(d==1){
        
        if(coords[index+n]>coords[Tree->index+n]){
            std::swap(temp1,temp2);
        }
        
        get_nn(temp1,index,(d+1)%P,coords,n, nnDist, nnIndx, iNNIndx, iNN,check);
        
        if(abs(coords[Tree->index+n]-coords[index+n])>nnDist[iNNIndx+iNN-1]){
            return;
        }
        
        get_nn(temp2,index,(d+1)%P,coords,n, nnDist, nnIndx, iNNIndx, iNN,check);
    }
    
}


void mkNNIndxTree0(int n, int m, double *coords, int *nnIndx, double *nnDist, int *nnIndxLU){
    
    int i, iNNIndx, iNN;
    double d;
    int nIndx = static_cast<int>(static_cast<double>(1+m)/2*m+(n-m-1)*m);
    int BUCKETSIZE = 10;
    
    
    for(i = 0; i < nIndx; i++){
        nnDist[i] = std::numeric_limits<double>::infinity();
    }
    
    Node *Tree=NULL;
    int time_through=-1;
    
    for(i=0;i<n;i++){
        getNNIndx(i, m, iNNIndx, iNN);
        nnIndxLU[i] = iNNIndx;
        nnIndxLU[n+i] = iNN;
        if(time_through==-1){
            time_through=i;
        }
        
        if(i!=0){
            for(int j = time_through; j < i; j++){
                getNNIndx(i, m, iNNIndx, iNN);
                d = dist2(coords[i], coords[i+n], coords[j], coords[n+j]);
                if(d < nnDist[iNNIndx+iNN-1]){
                    nnDist[iNNIndx+iNN-1] = d;
                    nnIndx[iNNIndx+iNN-1] = j;
                    
                    //fSort(&nnDist[iNNIndx], &nnIndx[iNNIndx], iNN);
                    rsort_with_index(&nnDist[iNNIndx], &nnIndx[iNNIndx], iNN);
                }
            }
            
            
            if(i%BUCKETSIZE==0){
                
#ifdef _OPENMP
#pragma omp parallel for private(iNNIndx, iNN)
#endif
                for(int j=time_through;j<time_through+BUCKETSIZE;j++){
                    
                    getNNIndx(j, m, iNNIndx, iNN);
                    get_nn(Tree,j,0, coords,n, nnDist,nnIndx,iNNIndx,iNN,i-BUCKETSIZE);
                }
                
                
                for(int j=time_through;j<time_through+BUCKETSIZE;j++){
                    Tree=miniInsert(Tree,coords,j,0, n);
                }
                
                time_through=-1;
            }
            if(i==n-1){
                
#ifdef _OPENMP
#pragma omp parallel for private(iNNIndx, iNN)
#endif
                for(int j=time_through;j<n;j++){
                    getNNIndx(j, m, iNNIndx, iNN);
                    get_nn(Tree,j,0, coords,n, nnDist,nnIndx,iNNIndx,iNN,i-BUCKETSIZE);
                }
                
            }
        }
        if(i==0){
            Tree=miniInsert(Tree,coords,i,0,n);
            time_through=-1;
        }
    }
    
    delete Tree;
}




extern "C" {

    //Global variables
    
    double *X_nngp;
    double *y_nngp;
    
    int n_nngp;
    int p_nngp;
    int m_nngp;
    
    int covModel_nngp;
    int nThreads_nngp;
    
    double *D_nngp;
    double *d_nngp;
    int *nnIndx_nngp;
    int *nnIndxLU_nngp;
    int *CIndx_nngp;
    
    
    int j_nngp;
    double eps_nngp;
    
    
    //covmodel = 0; exponential
    //covmodel = 1; spherical
    //covmodel = 2; matern
    //covmodel = 3; gaussian
    
    //Defining the likelihood (tausq/sigmasq = alphasq; phi = phi; nu = nu):
    
    
    //Update B and F:
    
    double updateBF(double *B, double *F, double *c, double *C, double *D, double *d, int *nnIndxLU, int *CIndx, int n, double *theta, int covModel, int nThreads){
        int i, k, l;
        int info = 0;
        int inc = 1;
        double one = 1.0;
        double zero = 0.0;
        char lower = 'L';
        double logDet = 0;
        double nu = 0;
        //check if the model is 'matern'
        if (covModel == 2) {
            nu = theta[2];
        }
        
        double *bk = (double *) R_alloc(nThreads*(static_cast<int>(1.0+5.0)), sizeof(double));
        
        
        //bk must be 1+(int)floor(alpha) * nthread
        int nb = 1+static_cast<int>(floor(5.0));
        int threadID = 0;
        
#ifdef _OPENMP
#pragma omp parallel for private(k, l, info, threadID)
#endif
        for(i = 0; i < n; i++){
#ifdef _OPENMP
            threadID = omp_get_thread_num();
#endif
            //theta[0] = alphasquareIndex, theta[1] = phiIndex, theta[2] = nuIndex (in case of 'matern')
            if(i > 0){
                for(k = 0; k < nnIndxLU[n+i]; k++){
                    c[nnIndxLU[i]+k] = spCor(d[nnIndxLU[i]+k], theta[1], nu, covModel, &bk[threadID*nb]);
                    for(l = 0; l <= k; l++){
                        C[CIndx[i]+l*nnIndxLU[n+i]+k] = spCor(D[CIndx[i]+l*nnIndxLU[n+i]+k], theta[1], nu, covModel, &bk[threadID*nb]);
                        if(l == k){
                            C[CIndx[i]+l*nnIndxLU[n+i]+k] += theta[0];
                        }
                    }
                }
                F77_NAME(dpotrf)(&lower, &nnIndxLU[n+i], &C[CIndx[i]], &nnIndxLU[n+i], &info); if(info != 0){error("c++ error: dpotrf failed\n");}
                F77_NAME(dpotri)(&lower, &nnIndxLU[n+i], &C[CIndx[i]], &nnIndxLU[n+i], &info); if(info != 0){error("c++ error: dpotri failed\n");}
                F77_NAME(dsymv)(&lower, &nnIndxLU[n+i], &one, &C[CIndx[i]], &nnIndxLU[n+i], &c[nnIndxLU[i]], &inc, &zero, &B[nnIndxLU[i]], &inc);
                F[i] = 1 - F77_NAME(ddot)(&nnIndxLU[n+i], &B[nnIndxLU[i]], &inc, &c[nnIndxLU[i]], &inc) + theta[0];
            }else{
                B[i] = 0;
                F[i] = 1 + theta[0];
            }
        }
        for(i = 0; i < n; i++){
            logDet += log(F[i]);
        }
        
        return(logDet);
    }
    
    void solve_B_F(double *B, double *F, double *norm_residual_boot, int n, int *nnIndxLU, int *nnIndx, double *residual_boot){
        
        residual_boot[0] = norm_residual_boot[0] * sqrt(F[0]);
        double sum;
        for (int i = 1; i < n; i++) {
            sum = norm_residual_boot[i];
            for (int l = 0; l < nnIndxLU[n + i]; l++) {
                sum = sum + B[nnIndxLU[i] + l] * residual_boot[nnIndx[nnIndxLU[i] + l]] / sqrt(F[i]);
            }
            residual_boot[i] = sum * sqrt(F[i]);
        }
    }
    
    
    
    void product_B_F(double *B, double *F, double *residual_nngp, int n, int *nnIndxLU, int *nnIndx, double *norm_residual_nngp){
        norm_residual_nngp[0] = residual_nngp[0]/sqrt(F[0]);
        double sum;
        for (int i = 1; i < n; i++) {
            sum = 0.0;
            for (int l = 0; l < nnIndxLU[n + i]; l++) {
                sum = sum - B[nnIndxLU[i] + l] * residual_nngp[nnIndx[nnIndxLU[i] + l]] / sqrt(F[i]);
            }
            norm_residual_nngp[i] = sum + residual_nngp[i] / sqrt(F[i]);
        }
    }
    
    
    void processed_output(double *X, double *y, double *D, double *d, int *nnIndx, int *nnIndxLU, int *CIndx, int n, int p, int m, double *theta, int covModel, int j, int nThreads, double optimized_likelihod, double *B, double *F, double *beta, double *Xbeta, double *norm_residual, double *theta_fp){
        
        char const *ntran = "N";
        int nIndx = static_cast<int>(static_cast<double>(1+m)/2*m+(n-m-1)*m);
        double *c =(double *) R_alloc(nIndx, sizeof(double));
        double *C = (double *) R_alloc(j, sizeof(double)); zeros(C, j);
        
        double logDet;
        
        int pp = p*p;
        int info = 0;
        const double negOne = -1.0;
        const double one = 1.0;
        const double zero = 0.0;
        const int inc = 1;
        char const *lower = "L";
        
        
        double *tmp_pp = (double *) R_alloc(pp, sizeof(double));
        double *tmp_p = (double *) R_alloc(p, sizeof(double));
        double *tmp_n = (double *) R_alloc(n, sizeof(double));
        double *residual = (double *) R_alloc(n, sizeof(double));
        
        //create B and F
        logDet = updateBF(B, F, c, C, D, d, nnIndxLU, CIndx, n, theta, covModel, nThreads);
        
        int i;
        for(i = 0; i < p; i++){
            tmp_p[i] = Q(B, F, &X[n*i], y, n, nnIndx, nnIndxLU);
            for(j = 0; j <= i; j++){
                tmp_pp[j*p+i] = Q(B, F, &X[n*j], &X[n*i], n, nnIndx, nnIndxLU);
            }
        }
        
        F77_NAME(dpotrf)(lower, &p, tmp_pp, &p, &info); if(info != 0){error("c++ error: dpotrf failed\n");}
        F77_NAME(dpotri)(lower, &p, tmp_pp, &p, &info); if(info != 0){error("c++ error: dpotri failed\n");}
        
        //create Beta
        F77_NAME(dsymv)(lower, &p, &one, tmp_pp, &p, tmp_p, &inc, &zero, beta, &inc);
        
        //create Xbeta
        F77_NAME(dgemv)(ntran, &n, &p, &one, X, &n, beta, &inc, &zero, tmp_n, &inc);
        
        dcopy_(&n, tmp_n, &inc, Xbeta, &inc);
        
        
        //create normalized residual
        F77_NAME(daxpy)(&n, &negOne, y, &inc, tmp_n, &inc);
        
        for (int s = 0; s < n; s++) {
            residual[s] = negOne * tmp_n[s];
        }
        
        product_B_F(B, F, residual, n, nnIndxLU, nnIndx, norm_residual);
        
        
        //Create complete theta
        
        // 1. Create sigma square
        theta_fp[0] = exp((optimized_likelihod - logDet)/n);
        
        
        // 2. Create tau square
        theta_fp[1] = theta[0] * theta_fp[0];
        
        // 3. Create phi
        theta_fp[2] = theta[1];
        
        // 4. Create nu in "matern"
        if (covModel == 2) {
            theta_fp[3] = theta[2];
        }
    }

    
    
    //Defining likelihood in terms of theta.
    double likelihood(double *X, double *y, double *D, double *d, int *nnIndx, int *nnIndxLU, int *CIndx, int n, int p, int m, double *theta, int covModel, int j, int nThreads){
        
        char const *ntran = "N";
        int nIndx = static_cast<int>(static_cast<double>(1+m)/2*m+(n-m-1)*m);
        double *B = (double *) R_alloc(nIndx, sizeof(double));
        double *F = (double *) R_alloc(n, sizeof(double));
        double *c =(double *) R_alloc(nIndx, sizeof(double));
        double *C = (double *) R_alloc(j, sizeof(double)); zeros(C, j);
        
        double logDet;
        
        int pp = p*p;
        int info = 0;
        double log_likelihood;
        const double negOne = -1.0;
        const double one = 1.0;
        const double zero = 0.0;
        const int inc = 1;
        char const *lower = "L";
        
        
        double *tmp_pp = (double *) R_alloc(pp, sizeof(double));
        double *tmp_p = (double *) R_alloc(p, sizeof(double));
        double *beta = (double *) R_alloc(p, sizeof(double));
        double *tmp_n = (double *) R_alloc(n, sizeof(double));
        
        logDet = updateBF(B, F, c, C, D, d, nnIndxLU, CIndx, n, theta, covModel, nThreads);
        
        int i;
        for(i = 0; i < p; i++){
            tmp_p[i] = Q(B, F, &X[n*i], y, n, nnIndx, nnIndxLU);
            for(j = 0; j <= i; j++){
                tmp_pp[j*p+i] = Q(B, F, &X[n*j], &X[n*i], n, nnIndx, nnIndxLU);
            }
        }
        
        F77_NAME(dpotrf)(lower, &p, tmp_pp, &p, &info); if(info != 0){error("c++ error: dpotrf failed\n");}
        F77_NAME(dpotri)(lower, &p, tmp_pp, &p, &info); if(info != 0){error("c++ error: dpotri failed\n");}
        F77_NAME(dsymv)(lower, &p, &one, tmp_pp, &p, tmp_p, &inc, &zero, beta, &inc);
        F77_NAME(dgemv)(ntran, &n, &p, &one, X, &n, beta, &inc, &zero, tmp_n, &inc);
        F77_NAME(daxpy)(&n, &negOne, y, &inc, tmp_n, &inc);
        
        //calculates likelihood
        log_likelihood = n * log(Q(B, F, tmp_n, tmp_n, n, nnIndx, nnIndxLU)/n) + logDet;
        
        return(log_likelihood);
    }
    
    
    
    //Defining likelihood w.r.t unconstrained optimization with alpha, root_phi, root_nu (in case of matern);
    
    // a. Non-matern models
    double likelihood_lbfgs_non_matern(double alpha, double root_phi, double *X, double *y, double *D, double *d, int *nnIndx, int *nnIndxLU, int *CIndx, int n, int p, int m, int covModel, int j, int nThreads){
        double *theta = (double *) R_alloc(2, sizeof(double));
        theta[0] = pow(alpha, 2.0);
        theta[1] = pow(root_phi, 2.0);
        double res = likelihood(X, y, D, d, nnIndx, nnIndxLU, CIndx, n, p, m, theta, covModel, j, nThreads);//some unnecessary checking are happening here, will remove afterwards
        return(res);
    }
    
    
    //b. matern models
    double likelihood_lbfgs_matern(double alpha, double root_phi, double root_nu, double *X, double *y, double *D, double *d, int *nnIndx, int *nnIndxLU, int *CIndx, int n, int p, int m, int covModel, int j, int nThreads){
        double *theta = (double *) R_alloc(3, sizeof(double));
        theta[0] = pow(alpha, 2.0);
        theta[1] = pow(root_phi, 2.0);
        theta[2] = pow(root_nu, 2.0);
        double res = likelihood(X, y, D, d, nnIndx, nnIndxLU, CIndx, n, p, m, theta, covModel, j, nThreads);//some unnecessary checking are happening here, will remove afterwards;
        return(res);
    }
    
    
    static lbfgsfloatval_t evaluate(
                                    void *instance,
                                    const lbfgsfloatval_t *x,
                                    lbfgsfloatval_t *g,
                                    const int n,
                                    const lbfgsfloatval_t step
                                    )
    {
        int i;
        lbfgsfloatval_t fx = 0.0;
        
        if (covModel_nngp != 2) {
            for (i = 0;i < n;i += 2) {
                g[i+1] = (likelihood_lbfgs_non_matern(x[i], (x[i+1]  + eps_nngp), X_nngp, y_nngp, D_nngp, d_nngp, nnIndx_nngp, nnIndxLU_nngp, CIndx_nngp, n_nngp, p_nngp, m_nngp, covModel_nngp, j_nngp, nThreads_nngp) - likelihood_lbfgs_non_matern(x[i], (x[i+1] - eps_nngp), X_nngp, y_nngp, D_nngp, d_nngp, nnIndx_nngp, nnIndxLU_nngp, CIndx_nngp, n_nngp, p_nngp, m_nngp, covModel_nngp, j_nngp, nThreads_nngp))/(2*eps_nngp);
                
                g[i] = (likelihood_lbfgs_non_matern((x[i] + eps_nngp), x[i+1], X_nngp, y_nngp, D_nngp, d_nngp, nnIndx_nngp, nnIndxLU_nngp, CIndx_nngp, n_nngp, p_nngp, m_nngp, covModel_nngp, j_nngp, nThreads_nngp) - likelihood_lbfgs_non_matern((x[i] - eps_nngp), x[i+1], X_nngp, y_nngp, D_nngp, d_nngp, nnIndx_nngp, nnIndxLU_nngp, CIndx_nngp, n_nngp, p_nngp, m_nngp, covModel_nngp, j_nngp, nThreads_nngp))/(2*eps_nngp);
                
                fx += likelihood_lbfgs_non_matern(x[i], x[i+1], X_nngp, y_nngp, D_nngp, d_nngp, nnIndx_nngp, nnIndxLU_nngp, CIndx_nngp, n_nngp, p_nngp, m_nngp, covModel_nngp,j_nngp, nThreads_nngp);
            }
        } else {
            for (i = 0;i < n;i += 3) {
                g[i+1] = (likelihood_lbfgs_matern(x[i], (x[i+1]  + eps_nngp), x[i+2], X_nngp, ::y_nngp, D_nngp, d_nngp, nnIndx_nngp, nnIndxLU_nngp, CIndx_nngp, n_nngp, p_nngp, m_nngp, covModel_nngp, j_nngp, nThreads_nngp) - likelihood_lbfgs_matern(x[i], (x[i+1] - eps_nngp), x[i+2], X_nngp, y_nngp, D_nngp, d_nngp, nnIndx_nngp, nnIndxLU_nngp, CIndx_nngp, n_nngp, p_nngp, m_nngp, covModel_nngp, j_nngp, nThreads_nngp))/(2*eps_nngp);
                
                g[i] = (likelihood_lbfgs_matern((x[i] + eps_nngp), x[i+1], x[i+2], X_nngp, ::y_nngp, D_nngp, d_nngp, nnIndx_nngp, nnIndxLU_nngp, CIndx_nngp, n_nngp, p_nngp, m_nngp, covModel_nngp, j_nngp, nThreads_nngp) - likelihood_lbfgs_matern((x[i] - eps_nngp), x[i+1], x[i+2], X_nngp, y_nngp, D_nngp, d_nngp, nnIndx_nngp, nnIndxLU_nngp, CIndx_nngp, n_nngp, p_nngp, m_nngp, covModel_nngp, j_nngp, nThreads_nngp))/(2*eps_nngp);
                
                g[i+2] = (likelihood_lbfgs_matern(x[i], x[i+1], (x[i+2] + eps_nngp), X_nngp, y_nngp, D_nngp, d_nngp, nnIndx_nngp, nnIndxLU_nngp, CIndx_nngp, n_nngp, p_nngp, m_nngp, covModel_nngp, j_nngp, nThreads_nngp) - likelihood_lbfgs_matern(x[i], x[i+1], (x[i+2] - eps_nngp), X_nngp, y_nngp, D_nngp, d_nngp, nnIndx_nngp, nnIndxLU_nngp, CIndx_nngp, n_nngp, p_nngp, m_nngp, covModel_nngp, j_nngp, nThreads_nngp))/(2*eps_nngp);
                
                fx += likelihood_lbfgs_matern(x[i], x[i+1], x[i+2], X_nngp, y_nngp, D_nngp, d_nngp, nnIndx_nngp, nnIndxLU_nngp, CIndx_nngp, n_nngp, p_nngp, m_nngp, covModel_nngp, j_nngp, nThreads_nngp);
            }
        }
        return fx;
    }
    
    void processed_bootstrap_output(double *X, double *y_boot, double *D, double *d, int *nnIndx, int *nnIndxLU, int *CIndx, int n, int p, int m, double *theta, int covModel, int j, int nThreads, double optimized_likelihod, double *beta_boot, double *theta_fp_boot){
        
        int nIndx = static_cast<int>(static_cast<double>(1+m)/2*m+(n-m-1)*m);
        double *B = (double *) R_alloc(nIndx, sizeof(double));
        double *F = (double *) R_alloc(n, sizeof(double));
        double *c =(double *) R_alloc(nIndx, sizeof(double));
        double *C = (double *) R_alloc(j, sizeof(double)); zeros(C, j);
        
        double logDet;
        
        int pp = p*p;
        int info = 0;
        const double negOne = -1.0;
        const double one = 1.0;
        const double zero = 0.0;
        const int inc = 1;
        char const *lower = "L";
        
        
        double *tmp_pp = (double *) R_alloc(pp, sizeof(double));
        double *tmp_p = (double *) R_alloc(p, sizeof(double));
        double *tmp_n = (double *) R_alloc(n, sizeof(double));
        
        //create B and F
        logDet = updateBF(B, F, c, C, D, d, nnIndxLU, CIndx, n, theta, covModel, nThreads);
        
        int i;
        for(i = 0; i < p; i++){
            tmp_p[i] = Q(B, F, &X[n*i], y_boot, n, nnIndx, nnIndxLU);
            for(j = 0; j <= i; j++){
                tmp_pp[j*p+i] = Q(B, F, &X[n*j], &X[n*i], n, nnIndx, nnIndxLU);
            }
        }
        
        F77_NAME(dpotrf)(lower, &p, tmp_pp, &p, &info); if(info != 0){error("c++ error: dpotrf failed\n");}
        F77_NAME(dpotri)(lower, &p, tmp_pp, &p, &info); if(info != 0){error("c++ error: dpotri failed\n");}
        
        //create Beta
        F77_NAME(dsymv)(lower, &p, &one, tmp_pp, &p, tmp_p, &inc, &zero, beta_boot, &inc);
        
        
        //create residual
        F77_NAME(daxpy)(&n, &negOne, y_boot, &inc, tmp_n, &inc);
        
        
        
        //Create complete theta
        
        // 1. Create sigma square
        theta_fp_boot[0] = exp((optimized_likelihod - logDet)/n);
        
        
        // 2. Create tau square
        theta_fp_boot[1] = theta[0] * theta_fp_boot[0];
        
        // 3. Create phi
        theta_fp_boot[2] = theta[1];
        
        // 4. Create nu in "matern"
        if (covModel == 2) {
            theta_fp_boot[3] = theta[2];
        }
    }
    
    SEXP process_bootstrap_data(SEXP B_r, SEXP F_r, SEXP Xbeta_r, SEXP norm_residual_boot_r, SEXP nnIndx_r, SEXP nnIndxLU_r, SEXP n_r, SEXP p_r){
        const int inc = 1;
        const double one = 1.0;
        n_nngp = INTEGER(n_r)[0];
        nnIndxLU_nngp = INTEGER(nnIndxLU_r);
        nnIndx_nngp = INTEGER(nnIndx_r);
        
        int nProtect = 0;
        
        SEXP residual_boot_r; PROTECT(residual_boot_r = allocVector(REALSXP, n_nngp)); nProtect++; double *residual_boot = REAL(residual_boot_r);
        
        solve_B_F(REAL(B_r), REAL(F_r), REAL(norm_residual_boot_r), n_nngp, INTEGER(nnIndxLU_r), INTEGER(nnIndx_r), residual_boot);
        
        //create y corresponding to boot
        F77_NAME(daxpy)(&n_nngp, &one, REAL(Xbeta_r), &inc, residual_boot, &inc);
        
        SEXP result_r, resultName_r;
        int nResultListObjs = 1;
        
        PROTECT(result_r = allocVector(VECSXP, nResultListObjs)); nProtect++;
        PROTECT(resultName_r = allocVector(VECSXP, nResultListObjs)); nProtect++;
        
        SET_VECTOR_ELT(result_r, 0, residual_boot_r);
        SET_VECTOR_ELT(resultName_r, 0, mkChar("result"));
        
        
        namesgets(result_r, resultName_r);
        //unprotect
        UNPROTECT(nProtect);
        
        
        return(result_r);
    }
    
    
    SEXP process_bootstrap(SEXP X_r, SEXP B_r, SEXP F_r, SEXP Xbeta_r, SEXP norm_residual_boot_r, SEXP D_r, SEXP d_r, SEXP nnIndx_r, SEXP nnIndxLU_r, SEXP CIndx_r, SEXP n_r, SEXP p_r, SEXP m_r, SEXP theta_r, SEXP covModel_r, SEXP j_r, SEXP nThreads_r, SEXP eps_r){
        
        const int inc = 1;
        const double one = 1.0;
        X_nngp = REAL(X_r);
        p_nngp = INTEGER(p_r)[0];
        n_nngp = INTEGER(n_r)[0];
        double *theta_start = REAL(theta_r);
        D_nngp = REAL(D_r);
        d_nngp = REAL(d_r);
        nnIndxLU_nngp = INTEGER(nnIndxLU_r);
        nnIndx_nngp = INTEGER(nnIndx_r);
        CIndx_nngp = INTEGER(CIndx_r);
        j_nngp = INTEGER(j_r)[0];
        covModel_nngp = INTEGER(covModel_r)[0];
        nThreads_nngp = INTEGER(nThreads_r)[0];
        m_nngp = INTEGER(m_r)[0];
        eps_nngp = REAL(eps_r)[0];
        
        
        int nProtect = 0;
        
        SEXP residual_boot_r; PROTECT(residual_boot_r = allocVector(REALSXP, n_nngp)); nProtect++; double *residual_boot = REAL(residual_boot_r);
        
        solve_B_F(REAL(B_r), REAL(F_r), REAL(norm_residual_boot_r), n_nngp, INTEGER(nnIndxLU_r), INTEGER(nnIndx_r), residual_boot);
        
        //create y corresponding to boot
        F77_NAME(daxpy)(&n_nngp, &one, REAL(Xbeta_r), &inc, residual_boot, &inc);
        
        y_nngp = residual_boot;
        
        int nTheta;
        
        if(covModel_nngp != 2){
            nTheta = 2;//tau^2 = 0, phi = 1
        }else{
            nTheta = 3;//tau^2 = 0, phi = 1, nu = 2;
        }
        
        int i_0, ret = 0;
        int k_0 = 0;
        lbfgsfloatval_t fx;
        lbfgsfloatval_t *x = lbfgs_malloc(nTheta);
        lbfgs_parameter_t param;
        
        /* Initialize the variables. */
        for (i_0 = 0;i_0 < nTheta; i_0++) {
            x[i_0] = theta_start[i_0];
        }
        
        /* Initialize the parameters for the L-BFGS optimization. */
        lbfgs_parameter_init(&param);
        param.epsilon = 1e-2;
        param.gtol = 0.9;
        /*param.linesearch = LBFGS_LINESEARCH_BACKTRACKING;*/
        
        /*
         Start the L-BFGS optimization; this will invoke the callback functions
         evaluate() and progress() when necessary.
         */
        ret = lbfgs(nTheta, x, &fx, evaluate, NULL, NULL, &param);
        
        // Construct output
        double *theta_boot = (double *) R_alloc(nTheta, sizeof(double));
        for (k_0 = 0; k_0 < nTheta; k_0++){
            theta_boot[k_0] = pow(x[k_0], 2.0);
        }
        
        // Clean up
        lbfgs_free(x);
        
        int nTheta_full = nTheta + 1;
        
        SEXP theta_fp_r; PROTECT(theta_fp_r = allocVector(REALSXP, nTheta_full)); nProtect++; double *theta_fp_boot = REAL(theta_fp_r);
        
        SEXP beta_r; PROTECT(beta_r = allocVector(REALSXP, p_nngp)); nProtect++; double *beta_boot = REAL(beta_r);
        
        processed_bootstrap_output(X_nngp, y_nngp, D_nngp, d_nngp, nnIndx_nngp, nnIndxLU_nngp, CIndx_nngp, n_nngp, p_nngp, m_nngp, theta_boot, covModel_nngp, j_nngp, nThreads_nngp, fx, beta_boot, theta_fp_boot);

        SEXP result_r, resultName_r;
        int nResultListObjs = 2;
        
        PROTECT(result_r = allocVector(VECSXP, nResultListObjs)); nProtect++;
        PROTECT(resultName_r = allocVector(VECSXP, nResultListObjs)); nProtect++;
        
        SET_VECTOR_ELT(result_r, 0, theta_fp_r);
        SET_VECTOR_ELT(resultName_r, 0, mkChar("theta"));
        
        SET_VECTOR_ELT(result_r, 1, beta_r);
        SET_VECTOR_ELT(resultName_r, 1, mkChar("Beta"));
        
        namesgets(result_r, resultName_r);
        //unprotect
        UNPROTECT(nProtect);
        
        
        return(result_r);
    }

    
    
    SEXP nngp_boot(SEXP y_r, SEXP X_r, SEXP p_r, SEXP n_r, SEXP m_r, SEXP coords_r, SEXP covModel_r, SEXP alphaSqStarting_r, SEXP phiStarting_r, SEXP nuStarting_r,
                   SEXP n_boot_r, SEXP sType_r, SEXP returnNNIndx_r, SEXP nThreads_r, SEXP verbose_r, SEXP eps_r){
        int i, k, l, nProtect=0;
        
        //get args
        y_nngp = REAL(y_r);
        X_nngp = REAL(X_r);
        p_nngp = INTEGER(p_r)[0];
        n_nngp = INTEGER(n_r)[0];
        m_nngp = INTEGER(m_r)[0];
        eps_nngp = REAL(eps_r)[0];
        double *coords = REAL(coords_r);
        
        covModel_nngp = INTEGER(covModel_r)[0];
        std::string corName = getCorName(covModel_nngp);
        
        nThreads_nngp = INTEGER(nThreads_r)[0];
        int verbose = INTEGER(verbose_r)[0];
        
        
        
#ifdef _OPENMP
        omp_set_num_threads(nThreads_nngp);
         printf("We have Open MP!\n")
#else
        if(nThreads_nngp > 1){
            warning("n.omp.threads > %i, but source not compiled with OpenMP support.", nThreads_nngp);
            nThreads_nngp = 1;
        }
#endif
        
        if(verbose){
            Rprintf("----------------------------------------\n");
            Rprintf("\tModel description\n");
            Rprintf("----------------------------------------\n");
            Rprintf("NNGP Response model fit with %i observations.\n\n", n_nngp);
            Rprintf("Number of covariates %i (including intercept if specified).\n\n", p_nngp);
            Rprintf("Using the %s spatial correlation model.\n\n", corName.c_str());
            Rprintf("Using %i nearest neighbors.\n\n", m_nngp);
#ifdef _OPENMP
            Rprintf("\nSource compiled with OpenMP support and model fit using %i thread(s).\n", nThreads_nngp);
#else
            Rprintf("\n\nSource not compiled with OpenMP support.\n");
#endif
        }
        
        //parameters
        int nTheta;
        
        if(corName != "matern"){
            nTheta = 2;//tau^2 = 0, phi = 1
        }else{
            nTheta = 3;//tau^2 = 0, phi = 1, nu = 2;
        }
        //starting
        double *theta = (double *) R_alloc (nTheta, sizeof(double));
        
        theta[0] = REAL(alphaSqStarting_r)[0];
        theta[1] = REAL(phiStarting_r)[0];
        
        if(corName == "matern"){
            theta[2] = REAL(nuStarting_r)[0];
        }
        
        //allocated for the nearest neighbor index vector (note, first location has no neighbors).
        int nIndx = static_cast<int>(static_cast<double>(1+m_nngp)/2*m_nngp+(n_nngp-m_nngp-1)*m_nngp);
        SEXP nnIndx_r; PROTECT(nnIndx_r = allocVector(INTSXP, nIndx)); nProtect++; nnIndx_nngp = INTEGER(nnIndx_r);
        SEXP d_r; PROTECT(d_r = allocVector(REALSXP, nIndx)); nProtect++; d_nngp = REAL(d_r);
        
        SEXP nnIndxLU_r; PROTECT(nnIndxLU_r = allocVector(INTSXP, 2*n_nngp)); nProtect++; nnIndxLU_nngp = INTEGER(nnIndxLU_r); //first column holds the nnIndx index for the i-th location and the second columns holds the number of neighbors the i-th location has (the second column is a bit of a waste but will simplifying some parallelization).
        
        //make the neighbor index
        if(verbose){
            Rprintf("----------------------------------------\n");
            Rprintf("\tBuilding neighbor index\n");
#ifdef Win32
            R_FlushConsole();
#endif
        }
        
        if(INTEGER(sType_r)[0] == 0){
            mkNNIndx(n_nngp, m_nngp, coords, nnIndx_nngp, d_nngp, nnIndxLU_nngp);
        }else{
            mkNNIndxTree0(n_nngp, m_nngp, coords, nnIndx_nngp, d_nngp, nnIndxLU_nngp);
        }
        
        
        SEXP CIndx_r; PROTECT(CIndx_r = allocVector(INTSXP, 2*n_nngp)); nProtect++; CIndx_nngp = INTEGER(CIndx_r); //index for D and C.
        for(i = 0, j_nngp = 0; i < n_nngp; i++){//zero should never be accessed
            j_nngp += nnIndxLU_nngp[n_nngp+i]*nnIndxLU_nngp[n_nngp+i];
            if(i == 0){
                CIndx_nngp[n_nngp+i] = 0;
                CIndx_nngp[i] = 0;
            }else{
                CIndx_nngp[n_nngp+i] = nnIndxLU_nngp[n_nngp+i]*nnIndxLU_nngp[n_nngp+i];
                CIndx_nngp[i] = CIndx_nngp[n_nngp+i-1] + CIndx_nngp[i-1];
            }
        }
        
        SEXP j_r; PROTECT(j_r = allocVector(INTSXP, 1)); nProtect++; INTEGER(j_r)[0] = j_nngp;
        
        SEXP D_r; PROTECT(D_r = allocVector(REALSXP, j_nngp)); nProtect++; D_nngp = REAL(D_r);
        
        for(i = 0; i < n_nngp; i++){
            for(k = 0; k < nnIndxLU_nngp[n_nngp+i]; k++){
                for(l = 0; l <= k; l++){
                    D_nngp[CIndx_nngp[i]+l*nnIndxLU_nngp[n_nngp+i]+k] = dist2(coords[nnIndx_nngp[nnIndxLU_nngp[i]+k]], coords[n_nngp+nnIndx_nngp[nnIndxLU_nngp[i]+k]], coords[nnIndx_nngp[nnIndxLU_nngp[i]+l]], coords[n_nngp+nnIndx_nngp[nnIndxLU_nngp[i]+l]]);
                }
            }
        }
        int i_0, ret = 0;
        int k_0 = 0;
        lbfgsfloatval_t fx;
        lbfgsfloatval_t *x = lbfgs_malloc(nTheta);
        lbfgs_parameter_t param;
        
        /* Initialize the variables. */
        for (i_0 = 0;i_0 < nTheta; i_0++) {
            x[i_0] = theta[i_0];
        }
        
        /* Initialize the parameters for the L-BFGS optimization. */
        lbfgs_parameter_init(&param);
        param.epsilon = 1e-2;
        param.gtol = 0.9;
        /*param.linesearch = LBFGS_LINESEARCH_BACKTRACKING;*/
        
        /*
         Start the L-BFGS optimization; this will invoke the callback functions
         evaluate() and progress() when necessary.
         */
        ret = lbfgs(nTheta, x, &fx, evaluate, NULL, NULL, &param);
        
        // Construct output
        double *theta_nngp = (double *) R_alloc(nTheta, sizeof(double));
        for (k_0 = 0; k_0 < nTheta; k_0++){
            theta_nngp[k_0] = pow(x[k_0], 2.0);
        }
        
        // Clean up
        lbfgs_free(x);
        
        int nTheta_full = nTheta + 1;
        
        SEXP B_r; PROTECT(B_r = allocVector(REALSXP, nIndx)); nProtect++; double *B_nngp = REAL(B_r);
        
        SEXP F_r; PROTECT(F_r = allocVector(REALSXP, n_nngp)); nProtect++; double *F_nngp = REAL(F_r);
        
        SEXP beta_r; PROTECT(beta_r = allocVector(REALSXP, p_nngp)); nProtect++; double *beta_nngp = REAL(beta_r);
        
        
        SEXP Xbeta_r; PROTECT(Xbeta_r = allocVector(REALSXP, n_nngp)); nProtect++; double *Xbeta_nngp = REAL(Xbeta_r);
        
        SEXP norm_residual_r; PROTECT(norm_residual_r = allocVector(REALSXP, n_nngp)); nProtect++; double *norm_residual_nngp = REAL(norm_residual_r);
        
        SEXP theta_fp_r; PROTECT(theta_fp_r = allocVector(REALSXP, nTheta_full)); nProtect++; double *theta_fp_nngp = REAL(theta_fp_r);

        processed_output(X_nngp, y_nngp, D_nngp, d_nngp, nnIndx_nngp, nnIndxLU_nngp, CIndx_nngp, n_nngp, p_nngp, m_nngp, theta_nngp, covModel_nngp, j_nngp, nThreads_nngp, fx, B_nngp, F_nngp, beta_nngp, Xbeta_nngp, norm_residual_nngp, theta_fp_nngp);
        
        //return stuff
        SEXP result_r, resultName_r;
        int nResultListObjs = 12;
        
        
        
        PROTECT(result_r = allocVector(VECSXP, nResultListObjs)); nProtect++;
        PROTECT(resultName_r = allocVector(VECSXP, nResultListObjs)); nProtect++;
        
        SET_VECTOR_ELT(result_r, 0, B_r);
        SET_VECTOR_ELT(resultName_r, 0, mkChar("B"));
        
        SET_VECTOR_ELT(result_r, 1, F_r);
        SET_VECTOR_ELT(resultName_r, 1, mkChar("F"));
        
        SET_VECTOR_ELT(result_r, 2, beta_r);
        SET_VECTOR_ELT(resultName_r, 2, mkChar("Beta"));
        
        SET_VECTOR_ELT(result_r, 3, norm_residual_r);
        SET_VECTOR_ELT(resultName_r, 3, mkChar("norm.residual"));
        
        SET_VECTOR_ELT(result_r, 4, theta_fp_r);
        SET_VECTOR_ELT(resultName_r, 4, mkChar("theta"));
        
        
        SET_VECTOR_ELT(result_r, 5, Xbeta_r);
        SET_VECTOR_ELT(resultName_r, 5, mkChar("Xbeta"));
        
        SET_VECTOR_ELT(result_r, 6, nnIndxLU_r);
        SET_VECTOR_ELT(resultName_r, 6, mkChar("nnIndxLU"));
        
        SET_VECTOR_ELT(result_r, 7, CIndx_r);
        SET_VECTOR_ELT(resultName_r, 7, mkChar("CIndx"));
        
        SET_VECTOR_ELT(result_r, 8, D_r);
        SET_VECTOR_ELT(resultName_r, 8, mkChar("D"));
        
        SET_VECTOR_ELT(result_r, 9, d_r);
        SET_VECTOR_ELT(resultName_r, 9, mkChar("d"));
        
        SET_VECTOR_ELT(result_r, 10, nnIndx_r);
        SET_VECTOR_ELT(resultName_r, 10, mkChar("nnIndx"));
        
        SET_VECTOR_ELT(result_r, 11, j_r);
        SET_VECTOR_ELT(resultName_r, 11, mkChar("Length.D"));
        
        
        namesgets(result_r, resultName_r);
        
        //unprotect
        UNPROTECT(nProtect);
        
        
        return(result_r);
    }
}




