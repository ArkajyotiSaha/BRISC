BRISC_simulation <- function(coords, sim_number = 1, seeds =  NULL, sigma.sq = 1, tau.sq = 0, phi = 1, nu = 1.5, n.neighbors = NULL, n_omp = 1,
                             cov.model = "exponential", search.type = "tree", stabilization = NULL, verbose = TRUE, tol = 12
){
  n <- nrow(coords)
  coords <- round(coords, tol)
  if(is.null(seeds)){
    sim <- matrix(rnorm(n*sim_number),n,sim_number)
  }
  if(!is.null(seeds)){
    if(length(seeds)!=sim_number){stop("error: number of seeds must be equal to number of simulations")}
    sim <- matrix(0,n,sim_number)
    for(sim_iter in 1:sim_number){
      set.seed(seeds[sim_iter])
      sim[,sim_iter] <- rnorm(n)
    }
  }

  result_list <- BRISC_correlation(coords, sim, sigma.sq, tau.sq, phi, nu, n.neighbors, n_omp, cov.model, search.type, stabilization, verbose, tol)
  return(result_list)
}
