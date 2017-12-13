brisc <- function(coords, x, y, sigma.sq = 1, tau.sq = 0.1, phi = 1, n.neighbors = 15, n_boot = 100, h = 1, n_omp = 1){
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
  if(h > 1){
    cl <- makeCluster(h)
    clusterExport(cl=cl, varlist=c("norm.residual", "X_data", "B", "F", "Xbeta", "D", "d", "nnIndx", "nnIndxLU", "CIndx", "n", "p", "n.neighbors", "theta_new_not", "cov.model.indx", "Length.D", "n.omp.threads", "bootstrap_brisc"),envir=environment())
    result <- parLapply(cl,1:n_boot,bootstrap_brisc,out$norm.residual, X_data, B, F, Xbeta, D, d, nnIndx, nnIndxLU, CIndx, n, p, n.neighbors, theta_new_not, cov.model.indx, Length.D, n.omp.threads)
    stopCluster(cl)
  }
  if(h == 1){
    result <- lapply(1:n_boot,bootstrap_brisc,out$norm.residual, X_data, B, F, Xbeta, D, d, nnIndx, nnIndxLU, CIndx, n, p, n.neighbors, theta_new_not, cov.model.indx, Length.D, n.omp.threads)
  }

  result_table = arrange(result)

  estimate <- c(out$Beta, out$theta)
  result_CI <- matrix(0,2,length(estimate))

  for(i in 1:length(estimate)){
    result_CI[,i] <- 2*estimate[i] - quantile(result_table[,i], c(.975,.025))
  }

  result_list <- list ()
  result_list$ord <- ord
  result_list$coords <- coords
  result_list$y <- y
  result_list$X <- X
  result_list$n.neighbors <- n.neighbors
  result_list$cov.model <- cov.model
  result_list$cov.model.indx <- cov.model.indx
  result_list$estimated.theta <- out$theta
  names(result_list$estimated.theta) <- c("sigma.sq", "tau.sq", "phi")
  result_list$estimated.beta <- out$Beta
  result_list$boot.theta <- result_table[,(length(out$Beta) + 1):dim(result_table)[2]]
  colnames(result_list$boot.theta) <- c("sigma.sq", "tau.sq", "phi")
  result_list$boot.beta <- result_table[,1:length(out$Beta)]
  result_list$confidence.interval <- result_CI
  colnames(result_list$confidence.interval)[(length(out$Beta) + 1):dim(result_table)[2]] <-  c("sigma.sq", "tau.sq", "phi")
  for(i in 1:length(out$Beta)){
    name_beta <- paste0("beta_",i)
    colnames(result_list$confidence.interval)[i] <- name_beta
  }
  class(result_list) <- "lmboot"
  result_list
}
