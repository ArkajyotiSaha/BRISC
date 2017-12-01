#------------------------------------------------
#  Brisc.R
#  Create 12/01/2017
#  Goal: Performs BRISC - Estimate parameters in a spatial linear model and provides bootstrap conficence intervals, Exponential covariance function.
#  Requirements: Mac, Openmp, Package Liblbfgs
#  Author: Arkajyoti Saha
# Acknowledgement: Package "spNNGP"
#------------------------------------------------

rm(list = ls())
if (!require("pacman")) install.packages("pacman")
pacman::p_load(parallel)

### Shared oject ##
dyn.load("brisc.so")


## bootstrap ##
boot_lapply <- function(i, e, X_data, B, F, Xbeta, D, d, nnIndx, nnIndxLU, CIndx, n, p, n.neighbors, theta_not, cov.model.indx, Length.D, n.omp.threads){
  dyn.load("brisc.so")
  set.seed(i)
  boot_indices = sample(1:n,n,replace = TRUE)
  e_boot <- e[boot_indices]
  boot_statistics <- .Call("process_bootstrap", X_data, B, F, Xbeta, e_boot, D, d, nnIndx, nnIndxLU, CIndx, n, p, n.neighbors, theta_not, cov.model.indx, Length.D, n.omp.threads, 1e-06)
  if(cov.model.indx == 2){
    result = c(Beta = boot_statistics$Beta, Sigma.Sq = boot_statistics$theta[1], Tau.Sq = boot_statistics$theta[2], Phi = boot_statistics$theta[3], Nu = boot_statistics$theta[4])
  }
  if(cov.model.indx != 2){
    result = c(Beta = boot_statistics$Beta, Sigma.Sq = boot_statistics$theta[1], Tau.Sq = boot_statistics$theta[2], Phi = boot_statistics$theta[3])
  }
  result
}

eval_bootbart_const<-function(x_r){
  c_matrix=matrix(0,length(x_r),length(x_r[[1]]))
  for(i in 1:length(x_r)){
    c_matrix[i,]<-(x_r[[i]])
  }
  c_matrix
}


## Main code ##
brisc <- function(coords, x, y, sigma.sq = 1, tau.sq = 0.1, phi = 1, n_boot = 100, h = 3, n_omp = 1){
  ord <- order(coords[,1] + coords[,2])
  coords <- coords[ord,]
  X <- x[ord,,drop=FALSE]
  y <- y[ord]
  cov.model <- "exponential"
  p <- ncol(X)
  n <- nrow(X)
  
  ##Coords and ordering
  if(!is.matrix(coords)){stop("error: coords must n-by-2 matrix of xy-coordinate locations")}
  if(ncol(coords) != 2 || nrow(coords) != n){
    stop("error: either the coords have more than two columns or then number of rows is different than
         data used in the model formula")
  }
  
  storage.mode(y) <- "double"
  storage.mode(X) <- "double"
  storage.mode(p) <- "integer"
  storage.mode(n) <- "integer"
  storage.mode(coords) <- "double"
  
  cov.model.names <- c("exponential","spherical","matern","gaussian")
  cov.model.indx <- which(cov.model == cov.model.names)-1
  storage.mode(cov.model.indx) <- "integer"
  
  alpha.sq.starting <- tau.sq/sigma.sq
  phi.starting <- phi
  nu.starting <- 0.1
  
  storage.mode(alpha.sq.starting) <- "double"
  storage.mode(phi.starting) <- "double"
  storage.mode(nu.starting) <- "double"
  
  search.type.names <- c("brute", "tree")
  search.type <- "tree"
  
  if(!search.type %in% search.type.names){
    stop("error: specified search.type '",search.type,"' is not a valid option; choose from ", paste(search.type.names, collapse=", ", sep="") ,".")
  }
  
  search.type.indx <- which(search.type == search.type.names)-1
  return.neighbors = FALSE
  storage.mode(search.type.indx) <- "integer"
  storage.mode(return.neighbors) <- "integer"
  verbose = FALSE
  n.omp.threads <- as.integer(n_omp)
  n.neighbors <- 15
  storage.mode(n.omp.threads) <- "integer"
  storage.mode(n.neighbors) <- "integer"
  storage.mode(verbose) <- "integer"
  out <- .Call("nngp_boot", y, X, p, n, n.neighbors, coords, cov.model.indx, alpha.sq.starting, phi.starting, nu.starting, as.integer(1), search.type.indx, return.neighbors, n.omp.threads, verbose, 2e-05)
  X_data <- X
  norm.residual = out$norm.residual
  B =  out$B
  F = out$F
  Xbeta = out$Xbeta
  D = out$D
  d = out$d
  nnIndx = out$nnIndx
  nnIndxLU = out$nnIndxLU
  CIndx = out$CIndx
  Length.D = out$Length.D
  theta_new_not = c(out$theta[2]/out$theta[1], out$theta[3])
  cl <- makeCluster(h)
  clusterExport(cl=cl, varlist=c("norm.residual", "X_data", "B", "F", "Xbeta", "D", "d", "nnIndx", "nnIndxLU", "CIndx", "n", "p", "n.neighbors", "theta_new_not", "cov.model.indx", "Length.D", "n.omp.threads", "boot_lapply"),envir=environment())
  result <- parLapply(cl,1:n_boot,boot_lapply,out$norm.residual, X_data, B, F, Xbeta, D, d, nnIndx, nnIndxLU, CIndx, n, p, n.neighbors, theta_new_not, cov.model.indx, Length.D, n.omp.threads)
  stopCluster(cl)
  
  
  result_table = eval_bootbart_const(result)
  
  estimate <- c(out$Beta, out$theta)
  result_new_bs <- rep(0,2*(length(estimate)))
  result_old_bs <- rep(0,2*(length(estimate)))
  
  for(i in 1:length(estimate)){
    result_new_bs[(2*i - 1):(2*i)] <- 2*estimate[i] - quantile(result_table[,i], c(.975,.025))
  }
  
  
  for(i in 1:length(estimate)){
    result_old_bs[(2*i - 1):(2*i)] <- quantile(result_table[,i], c(.025,.975))
  }
  
  result_list <- list ()
  result_list$y <- y
  result_list$X <- X
  result_list$n.neighbors <- n.neighbors
  result_list$coords <- coords
  result_list$ord <- ord
  result_list$cov.model <- cov.model
  result_list$cov.model.indx <- cov.model.indx
  result_list$estimated.theta <- out$theta
  result_list$estimated.beta <- out$Beta
  result_list$boot.beta <- result_table[,1:length(out$Beta)]
  result_list$boot.theta <- result_table[,(length(out$Beta) + 1):dim(result_table)[2]]
  result_list$confidence.interval <- c(result_new_bs, result_old_bs)
  result_list$summary <- c(Beta = out$Beta, Theta = out$theta, New_CI = result_new_bs, Old_CI = result_old_bs)
  class(result_list) <- "lmboot"
  result_list
}





####EXAMPLE#######

rmvn <- function(n, mu=0, V = matrix(1)){
  p <- length(mu)
  if(any(is.na(match(dim(V),p))))
    stop("Dimension problem!")
  D <- chol(V)
  t(matrix(rnorm(n*p), ncol=p)%*%D + rep(mu,rep(n,p)))
}

set.seed(1)
n <- 1000
coords <- cbind(runif(n,0,1), runif(n,0,1))

x <- cbind(rnorm(n), rnorm(n))

sigma.sq = 1
phi = 5
tau.sq = 0.1

beta <- c(1,5)
B <- as.matrix(beta)
D <- as.matrix(dist(coords))
R <- exp(-phi*D)
w <- rmvn(1, rep(0,n), sigma.sq*R)

y <- rnorm(n, x%*%B + w, sqrt(tau.sq))

result <- brisc(coords, x, y, n_boot = 10)
result$estimated.theta ## Gives estimation of (sigma.sq, tau.sq, Phi)
result$estimated.beta ## Gives estimation of (Beta)