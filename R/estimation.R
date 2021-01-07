BRISC_estimation <- function(coords, y, x = NULL, sigma.sq = 1, tau.sq = 0.1, phi = 1, nu = 1.5, n.neighbors = 15, n_omp = 1,
                             order = "Sum_coords", cov.model = "exponential", search.type = "tree", stabilization = NULL,
                             pred.stabilization = 1e-8, verbose = TRUE, eps = 2e-05, nugget_status = 1, tol = 12
){
  n <- nrow(coords)
  if(is.null(x)){
    x <- matrix(1, nrow = n, ncol = 1)
  }
  p <- ncol(x)
  if(ncol(x) == 1) warning('The ordering of inputs x (covariates) and y (response) in BRISC_estimation has been changed BRISC 1.0.0 onwards.
Please check the new documentation with ?BRISC_estimation.')
  ##Coords and ordering
  if(nugget_status == 0){fix_nugget = 0}
  if(nugget_status == 1){fix_nugget = 1}
  if(!is.matrix(coords)){stop("error: coords must n-by-2 matrix of xy-coordinate locations")}
  if(order == "AMMD" && length(y) < 65){stop("error: Number of data points must be atleast 65 to use AMMD")}
  if(ncol(coords) != 2 || nrow(coords) != n){
    stop("error: either the coords have more than two columns or the number of rows is different than
         data used in the model formula")
  }
  if(is.matrix(y) & ncol(as.matrix(y)) > 1)
    {stop("error: BRISC handles only univariate response. You have used a y (response) with more than one column.
          The ordering of inputs x (covariates) and y (response) in BRISC_estimation has been changed BRISC 0.2.0 onwards.
          Did you mean x (covaiates) instead of y (response) by the multivariatte input?
          Please check the new documentation with ?BRISC_estimation.")

  }

  coords <- round(coords, tol)
  x <- round(x, tol)
  y <- round(y, tol)
  if(order != "AMMD" && order != "Sum_coords"){
    stop("error: Please insert a valid ordering scheme choice given by 1 or 2.")
  }

  if(tau.sq < 0 ){stop("error: tau.sq must be non-negative")}
  if(sigma.sq < 0 ){stop("error: sigma.sq must be non-negative")}
  if(phi < 0 ){stop("error: phi must be non-negative")}
  if(nu < 0 ){stop("error: nu must be non-negative")}

  if(verbose == TRUE){
    cat(paste(("----------------------------------------"), collapse="   "), "\n"); cat(paste(("\tOrdering Coordinates"), collapse="   "), "\n")
    }
  if(order == "AMMD"){ord <- orderMaxMinLocal(coords)}
  if(order == "Sum_coords"){ord <- order(coords[,1] + coords[,2])}
  coords <- coords[ord,]

  ##Input data and ordering
  X <- x[ord,,drop=FALSE]
  y <- y[ord]



  ##Covariance model
  cov.model.names <- c("exponential","spherical","matern","gaussian")
  cov.model.indx <- which(cov.model == cov.model.names) - 1
  storage.mode(cov.model.indx) <- "integer"


  ##Stabilization
  if(is.null(stabilization)){
    if(cov.model == "exponential"){
      stabilization = FALSE
    }
    if(cov.model != "exponential"){
      stabilization = TRUE
    }
  }

  if(!isTRUE(stabilization)){
    if(cov.model != "exponential"){
      warning('We recommend using stabilization for spherical, Matern and Gaussian covariance model')
    }
  }

  if(isTRUE(stabilization)){
    taus <- 10^-6 * sigma.sq
    artificial_noise <- sqrt(taus) * rnorm(n)
    y <- y + artificial_noise
    tau.sq <- tau.sq + taus
    fix_nugget <- 1
    if(verbose == TRUE) cat(paste(("----------------------------------------"), collapse="   "), "\n"); cat(paste(("\tUsing Stabilization"), collapse="   "), "\n"); cat(paste(("----------------------------------------"), collapse="   "), "\n")
  }


  ##Initial values
  if(cov.model!="matern"){
    initiate <- c(sigma.sq, tau.sq, phi)
    names(initiate) <- c("sigma.sq", "tau.sq", "phi")
  }
  else{
    initiate <- c(sigma.sq, tau.sq, phi, nu)
    names(initiate) <- c("sigma.sq", "tau.sq", "phi", "nu")}

  alpha.sq.starting <- sqrt(tau.sq/sigma.sq)
  phi.starting <- sqrt(phi)
  nu.starting <- sqrt(nu)

  storage.mode(alpha.sq.starting) <- "double"
  storage.mode(phi.starting) <- "double"
  storage.mode(nu.starting) <- "double"


  ##Search type
  search.type.names <- c("brute", "tree", "cb")
  if(!search.type %in% search.type.names){
    stop("error: specified search.type '",search.type,"' is not a valid option; choose from ", paste(search.type.names, collapse=", ", sep="") ,".")
  }
  search.type.indx <- which(search.type == search.type.names)-1
  storage.mode(search.type.indx) <- "integer"


  ##Option for Multithreading if compiled with OpenMp support
  n.omp.threads <- as.integer(n_omp)
  storage.mode(n.omp.threads) <- "integer"


  ##type conversion
  storage.mode(y) <- "double"
  storage.mode(X) <- "double"
  storage.mode(p) <- "integer"
  storage.mode(n) <- "integer"
  storage.mode(coords) <- "double"
  storage.mode(n.neighbors) <- "integer"
  storage.mode(verbose) <- "integer"
  storage.mode(eps) <- "double"



  p1<- proc.time()

  if(nugget_status == 0){alpha.sq.starting = 0}


  ##estimtion
  result <- .Call("BRISC_estimatecpp", y, X, p, n, n.neighbors, coords, cov.model.indx, alpha.sq.starting, phi.starting, nu.starting, search.type.indx, n.omp.threads, verbose, eps, fix_nugget, PACKAGE = "BRISC")

  p2 <- proc.time()

  Theta <- result$theta
  if(!is.null(pred.stabilization)){
    Theta[2] <- max(result$theta[1] * pred.stabilization, result$theta[2])
  }
  if(cov.model!="matern"){
    names(Theta) <- c("sigma.sq", "tau.sq", "phi")
  }
  else{names(Theta) <- c("sigma.sq", "tau.sq", "phi", "nu")}

  Beta <- result$Beta
  for(i in 1:length(result$Beta)){
    name_beta <- paste0("beta_",i)
    names(Beta)[i] <- name_beta
  }

  result_list <- list ()
  result_list$ord <- ord
  result_list$coords <- coords
  result_list$y <- y
  result_list$X <- X
  result_list$n.neighbors <- n.neighbors
  result_list$cov.model <- cov.model
  result_list$eps <- eps
  result_list$init <- initiate
  result_list$Beta <- Beta
  result_list$Theta <- Theta
  result_list$estimation.time <- p2 - p1
  result_list$BRISC_Object <- result
  class(result_list) <- "BRISC_Out"
  result_list
}
