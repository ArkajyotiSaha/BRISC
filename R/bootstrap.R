BRISC_bootstrap <- function(BRISC_Out, n_boot = 100, h = 1, n_omp = 1, init = "Initial", verbose = TRUE){

  if(missing(BRISC_Out)){stop("error: BRISC_bootstrap expects BRISC_Out\n")}


  X <- BRISC_Out$X
  n.omp.threads <- as.integer(n_omp)
  n.neighbors <- BRISC_Out$n.neighbors
  eps <- BRISC_Out$eps
  cov.model <- BRISC_Out$cov.model
  p  <- ncol(X)
  n <- nrow(X)


  storage.mode(X) <- "double"
  storage.mode(p) <- "integer"
  storage.mode(n) <- "integer"
  storage.mode(n.neighbors) <- "integer"
  storage.mode(n.omp.threads) <- "integer"
  storage.mode(eps) <- "double"

  cov.model.names <- c("exponential","spherical","matern","gaussian")
  cov.model.indx <- which(cov.model == cov.model.names) - 1
  storage.mode(cov.model.indx) <- "integer"


  cov.model <- BRISC_Out$cov.model
  norm.residual = BRISC_Out$BRISC_Object$norm.residual
  B =  BRISC_Out$BRISC_Object$B
  F = BRISC_Out$BRISC_Object$F
  Xbeta = BRISC_Out$BRISC_Object$Xbeta
  D = BRISC_Out$BRISC_Object$D
  d = BRISC_Out$BRISC_Object$d
  nnIndx = BRISC_Out$BRISC_Object$nnIndx
  nnIndxLU = BRISC_Out$BRISC_Object$nnIndxLU
  CIndx = BRISC_Out$BRISC_Object$CIndx
  Length.D = BRISC_Out$BRISC_Object$Length.D

  if(init == "Initial"){
    if(cov.model == "matern") {theta_boot_init <- c(BRISC_Out$init[2]/BRISC_Out$init[1], BRISC_Out$init[3], BRISC_Out$init[4])}
    else {theta_boot_init <- c(BRISC_Out$init[2]/BRISC_Out$init[1], BRISC_Out$init[3])}
  }
  if(init == "Estimate"){
    if(cov.model == "matern") {theta_boot_init <- c(BRISC_Out$Theta[2]/BRISC_Out$Theta[1], BRISC_Out$Theta[3], BRISC_Out$Theta[4])}
    else {theta_boot_init <- c(BRISC_Out$Theta[2]/BRISC_Out$Theta[1], BRISC_Out$Theta[3])}
  }

  theta_boot_init <- sqrt(theta_boot_init)

  p3 <- proc.time()

  if(h > 1){
    cl <- makeCluster(h)
    clusterExport(cl=cl, varlist=c("norm.residual", "X", "B", "F", "Xbeta", "D", "d", "nnIndx", "nnIndxLU",
                                   "CIndx", "n", "p", "n.neighbors", "theta_boot_init", "cov.model.indx", "Length.D",
                                   "n.omp.threads", "bootstrap_brisc", "eps"),envir=environment())
    if(verbose == TRUE){
      cat(paste(("----------------------------------------"), collapse="   "), "\n"); cat(paste(("\tBootstrap Progress"), collapse="   "), "\n"); cat(paste(("----------------------------------------"), collapse="   "), "\n")
      pboptions(type = "txt", char = "=")
      result <- pblapply(1:n_boot,bootstrap_brisc,norm.residual, X, B, F, Xbeta, D, d, nnIndx, nnIndxLU, CIndx, n, p, n.neighbors, theta_boot_init,
                                            cov.model.indx, Length.D, n.omp.threads, eps, cl = cl)
      }
    if(verbose != TRUE){result <- parLapply(cl,1:n_boot,bootstrap_brisc,norm.residual, X, B, F, Xbeta, D, d, nnIndx, nnIndxLU, CIndx, n, p, n.neighbors, theta_boot_init,
                                            cov.model.indx, Length.D, n.omp.threads, eps)}
    stopCluster(cl)
  }
  if(h == 1){
    if(verbose == TRUE){
      cat(paste(("----------------------------------------"), collapse="   "), "\n"); cat(paste(("\tBootstrap Progress"), collapse="   "), "\n"); cat(paste(("----------------------------------------"), collapse="   "), "\n")
      pboptions(type = "txt", char = "=")
      result <- pblapply(1:n_boot,bootstrap_brisc,norm.residual, X, B, F, Xbeta, D, d, nnIndx, nnIndxLU, CIndx, n, p, n.neighbors, theta_boot_init,
                       cov.model.indx, Length.D, n.omp.threads, eps)
    }
    
    if(verbose != TRUE){
      result <- lapply(1:n_boot,bootstrap_brisc,norm.residual, X, B, F, Xbeta, D, d, nnIndx, nnIndxLU, CIndx, n, p, n.neighbors, theta_boot_init,
                       cov.model.indx, Length.D, n.omp.threads, eps)
    }
  }

  p4 <- proc.time()

  result_table = arrange(result)

  estimate <- c(BRISC_Out$Beta, BRISC_Out$Theta)
  result_CI <- matrix(0,2,length(estimate))

  for(i in 1:length(estimate)){
    result_CI[,i] <- 2*estimate[i] - quantile(result_table[,i], c(.975,.025))
  }

  result_list <- list()


  result_list$boot.Theta <- result_table[,(length(BRISC_Out$Beta) + 1):dim(result_table)[2]]
  if (cov.model != "matern") {colnames(result_list$boot.Theta) <- c("sigma.sq", "tau.sq", "phi")}
  if (cov.model == "matern") {colnames(result_list$boot.Theta) <- c("sigma.sq", "tau.sq", "phi", "nu")}
  result_list$boot.Beta <- as.matrix(result_table[,1:length(BRISC_Out$Beta)])
  colnames(result_list$boot.Beta) <- rep(0, length(BRISC_Out$Beta))
  for(i in 1:length(BRISC_Out$Beta)){
    name_beta <- paste0("beta_",i)
    colnames(result_list$boot.Beta)[i] <- name_beta
  }
  result_list$confidence.interval <- cbind(result_CI[,1:length(BRISC_Out$Beta)],pmax(result_CI[,(length(BRISC_Out$Beta) + 1)
                                     :dim(result_table)[2]], 0*result_CI[,(length(BRISC_Out$Beta) + 1):dim(result_table)[2]]))
  if (cov.model != "matern")  {colnames(result_list$confidence.interval)[(length(BRISC_Out$Beta) + 1):dim(result_table)[2]] <-
    c("sigma.sq", "tau.sq", "phi")}
  if (cov.model == "matern")  {colnames(result_list$confidence.interval)[(length(BRISC_Out$Beta) + 1):dim(result_table)[2]] <-
    c("sigma.sq", "tau.sq", "phi", "nu")}
  for(i in 1:length(BRISC_Out$Beta)){
    name_beta <- paste0("beta_",i)
    colnames(result_list$confidence.interval)[i] <- name_beta
  }
  result_list$boot.time = p4 - p3
  result_list
}
