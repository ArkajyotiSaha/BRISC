\name{BRISC_prediction}
\alias{BRISC_prediction}
\title{Function for performing prediction with BRISC}

\description{
  The function \code{BRISC_prediction} performs fast prediction on a set of new locations with univariate
  spatial regression models using Nearest Neighbor Gaussian Processes (NNGP) (Datta et al., 2016).
  BRISC_prediction  uses the parameter estimates from \code{BRISC_estimation} for the prediction. Some
  code blocks are borrowed from the R package: spNNGP: Spatial Regression Models for Large Datasets
  using Nearest Neighbor Gaussian Processes https://CRAN.R-project.org/package=spNNGP .
}

\usage{
BRISC_prediction(BRISC_Out, X.0, coords.0, n_omp = 1, verbose = TRUE)
}

\arguments{

  \item{BRISC_Out}{an object of class "BRISC_Out", obtained as an output of \code{BRISC_estimation}. }

  \item{X.0}{the covariates for prediction locations. Its Structure should be identical (including intercept) with that of
             covariates provided for estimation purpose in \code{BRISC_estimation}. }

  \item{coords.0}{the spatial coordinates corresponding to prediction locations. Its caleshould be same as that of coords
                  in \code{BRISC_estimation}. }

  \item{n_omp}{number of threads to be used, value can be more than 1 if source code is compiled with OpenMP support.
               Default is 1. }

  \item{verbose}{if TRUE, model specifications along with information regarding OpenMP support and progress of the
                 algorithm is printed to the screen. Otherwise, nothing is printed to the screen. Default value is FALSE. }

}

\value{
  A list comprising of the following:

  \item{prediction}{predicted response corresponding to X.0 and coords.0. }

  \item{prediction.ci}{confidence intervals corresponding to the predictions. }

   \item{prediction.time}{time (in seconds) required to perform the prediction after preprocessing data in R, reported using
                          \code{proc.time()}. }
}

\references{

  Datta, A., S. Banerjee, A.O. Finley, and A.E. Gelfand. (2016)
  Hierarchical Nearest-Neighbor Gaussian process models for large
  geostatistical datasets. Journal of the American Statistical
  Association, 111:800-812.

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

estimation_result <- BRISC_estimation(coords[1:800,], x[1:800,], y[1:800])
prediction_result <- BRISC_prediction(estimation_result, x[801:1000,], coords[801:1000,])

}

\keyword{model}
