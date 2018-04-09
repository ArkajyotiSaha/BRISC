BRISC_estimation <- function(coords, x, y, sigma.sq = 1, tau.sq = 0.1, phi = 1, nu = 0.5, n.neighbors = 15, n_omp = 1,
                             order = "AMMD", cov.model = "exponential", search.type = "tree", verbose = TRUE, eps = 2e-05
){
  p <- ncol(x)
  n <- nrow(x)
  ##Coords and ordering
  if(!is.matrix(coords)){stop("error: coords must n-by-2 matrix of xy-coordinate locations")}
  if(ncol(coords) != 2 || nrow(coords) != n){
    stop("error: either the coords have more than two columns or then number of rows is different than
         data used in the model formula")
  }
  if(order != "AMMD" && order != "Sum_coords"){
    stop("error: Please insert a valid ordering scheme choice given by 1 or 2.")
  }
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
  search.type.names <- c("brute", "tree")
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


  ##estimtion
  result <- .Call("BRISC_estimatecpp", y, X, p, n, n.neighbors, coords, cov.model.indx, alpha.sq.starting, phi.starting, nu.starting, search.type.indx, n.omp.threads, verbose, eps)

  p2 <- proc.time()

  Theta <- result$theta
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
  result_list$Beta = Beta
  result_list$Theta = Theta
  result_list$estimation.time = p2 - p1
  result_list$BRISC_Object = result
  class(result_list) <- "BRISC_Out"
  result_list
}
