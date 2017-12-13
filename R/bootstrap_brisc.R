bootstrap_brisc <- function(i, e, X_data, B, F, Xbeta, D, d, nnIndx, nnIndxLU, CIndx, n, p, n.neighbors, theta_not, cov.model.indx, Length.D, n.omp.threads){
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
