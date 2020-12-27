bootstrap_brisc <- function(i, e, X_data, B, F, Xbeta, D, d, nnIndx, nnIndxLU, CIndx, n, p, n.neighbors, theta_not, cov.model.indx, Length.D, n.omp.threads, eps, fix_nugget){
  set.seed(i)
  boot_indices = sample(1:n,n,replace = TRUE)
  e_boot <- e[boot_indices]
  boot_statistics <- .Call("BRISC_bootstrapcpp", X_data, B, F, Xbeta, e_boot, D, d, nnIndx, nnIndxLU, CIndx, n, p, n.neighbors, theta_not, cov.model.indx, Length.D, n.omp.threads, eps, fix_nugget, PACKAGE = "BRISC")
  if(cov.model.indx == 2){
    result = c(Beta = boot_statistics$Beta, sigma.sq = boot_statistics$theta[1], tau.sq = boot_statistics$theta[2], phi = boot_statistics$theta[3], nu = boot_statistics$theta[4])
  }
  if(cov.model.indx != 2){
    result = c(Beta = boot_statistics$Beta, sigma.sq = boot_statistics$theta[1], tau.sq = boot_statistics$theta[2], phi = boot_statistics$theta[3])
  }
  result
}
