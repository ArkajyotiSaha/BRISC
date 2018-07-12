\name{BRISC_bootstrap}
\alias{BRISC_bootstrap}
\title{Function for performing bootstrap with BRISC}

\description{
  The function \code{BRISC_bootstrap} performs bootstrap to provide confidence intervals for parameters of univariate spatial
  regression models using outputs of \code{BRISC_estimation}. The details of the bootstrap method can be found in BRISC
  (Saha & Datta, 2018). The optimization is performed with C library of limited-memory BFGS libLBFGS: a library of
  Limited-memory Broyden-Fletcher-Goldfarb-Shanno (L-BFGS), http://www.chokkan.org/software/liblbfgs/ (Naoaki Okazaki).
  For user convenience the soure codes of the package libLBFGS are provided in the package. Some code blocks are borrowed
  from the R package: spNNGP: Spatial Regression Models for Large Datasets using Nearest Neighbor Gaussian Processes
  https://CRAN.R-project.org/package=spNNGP .
}

\usage{
BRISC_bootstrap(BRISC_Out, n_boot = 100, h = 1, n_omp = 1,
                init = "Initial", verbose = TRUE)
}

\arguments{

  \item{BRISC_Out}{an object of class "BRISC_Out", obtained as an output of \code{BRISC_estimation}. }

  \item{n_boot}{number of bootstrap samples. Default value is 100. }

  \item{h}{Number of core to be used in parallel computing setup for bootstrap samples. If h = 1, there is no parallelization.
           Default value is 1. }

  \item{n_omp}{number of threads to be used, value can be more than 1 if source code is compiled with OpenMP support.
               Default is 1. }

  \item{init}{Keyword that specifies initialization scheme to be used. Supported keywords are: "Initial" and "Estimate"
              for initialization of parameter values for bootstrap samples with initial values used in \code{BRISC_estimate}
              and estimated values of parameters in \code{BRISC_estimate} respectively. }

  \item{verbose}{if TRUE, model specifications along with information regarding OpenMP support and progress of the algorithm is
                 printed to the screen. Otherwise, nothing is printed to the screen. Default value is FALSE. }
}

\value{
  A list comprising of the following:

  \item{boot.Theta}{estimates of spatial covariance parameters corresponding to bootstrap samples. }

  \item{boot.Beta}{estimates of beta corresponding to bootstrap samples. }

  \item{confidence.interval}{confidence intervals corresponding to the parameters. }

   \item{boot.time}{time (in seconds) required to perform the bootstrapping after preprocessing data in R, reported using
                    \code{proc.time()}. }
}

\references{
  Saha, A., & Datta, A. (2018). BRISC: bootstrap for rapid inference on spatial
  covariances. Stat, e184, DOI: 10.1002/sta4.184.

  Okazaki N. libLBFGS: a library of Limited-memory Broyden-Fletcher-Goldfarb-Shanno
  (L-BFGS), http://www.chokkan.org/software/liblbfgs/ .

  Andrew Finley, Abhirup Datta and Sudipto Banerjee (2017). spNNGP: Spatial Regression Models for Large
  Datasets using Nearest Neighbor Gaussian Processes. R package version 0.1.1.
  https://CRAN.R-project.org/package=spNNGP

}

\author{
  Arkajyoti Saha \email{arkajyotisaha93@gmail.com}, \cr
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
n <- 300
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

estimation_result <- BRISC_estimation(coords, x, y)
bootstrap_result <- BRISC_bootstrap(estimation_result, n_boot = 10)

}

\keyword{model}
