\name{brisc}
\alias{brisc}
\title{Function for fitting BRISC}

\description{
  The function \code{brisc} fits Gaussian univariate spatial
  regression models and performs Bootstrap using Nearest Neighbor Gaussian Processes (NNGP).
}

\usage{
brisc(coords, x, y, sigma.sq = 1, tau.sq = 0.1, phi = 1, n.neighbors = 15, n_boot = 100, h = 1, n_omp = 1)
}

\arguments{

  \item{coords}{an \eqn{n \times 2}{n x 2} matrix of the observation
  coordinates in \eqn{R^2}{R^2} (e.g., easting and northing). }

  \item{x}{an \eqn{n \times p}{n x p} matrix of the covariates in the observation coordinates. }

  \item{y}{a \eqn{n}{n} length vector of response at the observed coordinates. }

  \item{sigma.sq}{starting value of sigma square. Default value is 1. }

  \item{tau.sq}{starting value of tau square. Default value is 0.1. }

  \item{phi}{starting value of phi. Default value is 1. }

  \item{n.neighbors}{number of neighbors used in the NNGP. Default value is 15.}

  \item{n_boot}{number of bootstrap samples. Default is 100.}

  \item{h}{number of cluster to be used. Default is 1.}

  \item{n_omp}{number of threads to be used. Default is 1.}
}

\value{
  An object of class \code{lmboot}, which is a list comprising:

  \item{ord}{the vector \code{order(coords[,1] + coords[,2])}, which is the vector of indices used to order data necessary for fitting the NNGP
    model.}

  \item{coords}{the matrix \code{coords[ord,]}. }

  \item{y}{the vector \code{y[ord]}. }

  \item{X}{the matrix \code{x[ord,,drop=FALSE]}. }

  \item{n.neighbors}{the used value of \code{n.neighbors}. }

  \item{cov.model}{the used covariance model. }

  \item{cov.model.indx}{index of the used covariance model. }

  \item{estimated.theta}{estimate of spatial covariance parameters. }

  \item{estimated.beta}{estimate of beta. }

  \item{boot.theta}{estimates of spatial covariance parameters corresponding to bootstrap samples. }

  \item{boot.beta}{estimates of beta corresponding to bootstrap samples. }

  \item{confidence.interval}{confidence interval corresponding to the parameters. }
}


\author{
  Arkajyoti Saha \email{asaha8@jhmi.edu}, \cr
  Abhirup Datta \email{abhidatta@jhu.edu}
}

\examples{

rmvn <- function(n, mu = 0, V = matrix(1)){
  p <- length(mu)
  if(any(is.na(match(dim(V),p))))
    stop("Dimension not right!")
  D <- chol(V)
  t(matrix(rnorm(n*p), ncol=p)\%*\%D + rep(mu,rep(n,p)))
}

set.seed(1)
n <- 1000
coords <- cbind(runif(n,0,1), runif(n,0,1))

beta <- c(1,5)
x <- cbind(rnorm(n), rnorm(n))

sigma.sq = 1
phi = 5
tau.sq = 0.1

B <- as.matrix(beta)
D <- as.matrix(dist(coords))
R <- exp(-phi*D)
w <- rmvn(1, rep(0,n), sigma.sq*R)

y <- rnorm(n, x\%*\%B + w, sqrt(tau.sq))

ptm1 <- proc.time()
result <- brisc(coords, x, y, n_boot = 10, n_omp = 1)
proc.time() - ptm1
result$estimated.theta ## Gives estimation of (sigma.sq, tau.sq, phi)
result$estimated.beta ## Gives estimation of (Beta)

}

\keyword{model}
