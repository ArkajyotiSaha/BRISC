BRISC_prediction <- function(BRISC_Out, coords.0, X.0 = NULL, n_omp = 1, verbose = TRUE, tol = 12){


  if(missing(BRISC_Out)){stop("error: BRISC_prediction expects BRISC_Out\n")}

  n <- nrow(coords.0)
  if(is.null(X.0)){
    X.0 <- matrix(1, nrow = n, ncol = 1)
  }

  coords.0 <- round(coords.0, tol)
  X.0 <- round(X.0, tol)

  X <- BRISC_Out$X
  y <- BRISC_Out$y
  n.neighbors <- BRISC_Out$n.neighbors
  coords <- BRISC_Out$coords

  n.omp.threads <- as.integer(n_omp)
  p <- ncol(X)
  Theta <- as.matrix(BRISC_Out$Theta)
  dim(Theta) <- c(length(BRISC_Out$Theta),1)
  Beta <- as.matrix(BRISC_Out$Beta)
  dim(Beta) <- c(length(BRISC_Out$Beta),1)

  cov.model <- BRISC_Out$cov.model
  cov.model.names <- c("exponential","spherical","matern","gaussian")
  cov.model.indx <- which(cov.model == cov.model.names) - 1


  ##check X.0 and coords.0
  if(missing(X.0)){stop("error: X.0 must be specified\n")}
  if(!any(is.data.frame(X.0), is.matrix(X.0))){stop("error: X.0 must be a data.frame or matrix\n")}
  if(ncol(X.0) != ncol(X)){ stop(paste("error: X.0 must have ",p," columns\n"))}

  if(missing(coords.0)){stop("error: coords.0 must be specified\n")}
  if(!any(is.data.frame(coords.0), is.matrix(coords.0))){stop("error: coords.0 must be a data.frame or matrix\n")}
  if(!ncol(coords.0) == 2){stop("error: coords.0 must have two columns\n")}

  q <- nrow(X.0)

  ##get nn indx
  if(verbose == TRUE){
    cat(paste(("----------------------------------------"), collapse="   "), "\n"); cat(paste(("\tGetting nearest neighbors\n\tfor prediction locations"), collapse="   "), "\n")
    }
  nn.indx.0 <- nn2(coords, coords.0, k=n.neighbors)$nn.idx-1

  storage.mode(X) <- "double"
  storage.mode(y) <- "double"
  storage.mode(coords) <- "double"
  storage.mode(n) <- "integer"
  storage.mode(p) <- "integer"
  storage.mode(n.neighbors) <- "integer"
  storage.mode(X.0) <- "double"
  storage.mode(coords.0) <- "double"
  storage.mode(q) <- "integer"
  storage.mode(Beta) <- "double"
  storage.mode(Theta) <- "double"
  storage.mode(cov.model.indx) <- "integer"
  storage.mode(nn.indx.0) <- "integer"
  storage.mode(n.omp.threads) <- "integer"
  storage.mode(verbose) <- "integer"

  p5 <- proc.time()

  out <- .Call("BRISC_predictioncpp", X, y, coords, n, p, n.neighbors, X.0, coords.0, q, nn.indx.0,
               Beta, Theta, cov.model.indx, n.omp.threads, verbose, PACKAGE = "BRISC")
  p6 <- proc.time()

  output <- list()
  output$prediction <- out$p.y.0[,1]

  result_new <- matrix(0,length(output$prediction),2)
  result_new[,1] <- out$p.y.0[,1] - 1.96 * out$var.y.0[,1]
  result_new[,2] <- out$p.y.0[,1] + 1.96 * out$var.y.0[,1]

  output$prediction.ci <- result_new
  output$prediction.time <- p6 - p5
  output
}
