\name{BRISC_correlation}
\alias{BRISC_correlation}
\title{Function for create correlated data with BRISC}

\description{
  The function \code{BRISC_correlation} creates correlated data (known structure) using Nearest Neighbor
  Gaussian Processes (NNGP). \code{BRISC_correlation} uses the sparse Cholesky representation of Vecchia’s
  likelihood developed in Datta et al., 2016. Some code blocks are borrowed from the R package: spNNGP:
  Spatial Regression Models for Large Datasets using Nearest Neighbor Gaussian Processes
  https://CRAN.R-project.org/package=spNNGP .
}

\usage{
BRISC_correlation(coords, sim, sigma.sq = 1, tau.sq = 0, phi = 1, nu = 0.5, n.neighbors = 15,
                  n_omp = 1, cov.model = "exponential", search.type = "tree", verbose = TRUE)
}

\arguments{

  \item{coords}{an \eqn{n \times 2}{n x 2} matrix of the observation
  coordinates in \eqn{R^2} (e.g., easting and northing). }

  \item{sim}{an \eqn{n \times k}{n x k} matrix of the \eqn{k}{k} many \eqn{n \times 1}{n x 1} vectors
  from which the correlated data are calculated (see Details below). }

  \item{sigma.sq}{starting value of sigma square. Default value is 1. }

  \item{tau.sq}{starting value of tau square. Default value is 0.1. }

  \item{phi}{starting value of phi. Default value is 1. }

  \item{nu}{starting value of nu, only required for matern covariance model. Default value is 0.5. }

  \item{n.neighbors}{number of neighbors used in the NNGP. Default value is 15. }

  \item{n_omp}{number of threads to be used, value can be more than 1 if source code is compiled with OpenMP support.
               Default is 1. }

  \item{cov.model}{Keyword that specifies the covariance function to be used in modelling the spatial dependence structure
                   among the observations. Supported keywords are: "exponential", "matern", "spherical", and "gaussian"
                   for exponential, matern, spherical and gaussian covariance function respectively. Default value is "exponential". }

  \item{search.type}{Keyword that specifies type of nearest neighbor search algorithm to be used. Supported keywords are:
                     "tree" and "brute". Both of them provide the same result, though "tree" should be faster.
                     Default value is "tree". }

  \item{verbose}{if TRUE, model specifications along with information regarding OpenMP support and progress of the algorithm
                 is printed to the screen. Otherwise, nothing is printed to the screen. Default value is FALSE. }

}

\details{
Denote \eqn{g} be the input \code{sim}. Let \eqn{S} be the precision matrix associated with the covariance model determined by the \eqn{cov.model} and model parameters. Then \code{BRISC_correlation} calculates \eqn{h}, where \eqn{h} is given as follows:
\deqn{
T^0.5 h = g
}
where, \eqn{T^0.5} is a sparse approximation of the cholesky factor \eqn{S^0.5} of the precision matrix \eqn{S}, obtained from NNGP.
}

\value{
  A list comprising of the following:

  \item{coords}{the matrix \code{coords}. }

  \item{n.neighbors}{the used value of \code{n.neighbors}. }

  \item{cov.model}{the used covariance model. }

  \item{Theta}{Parameters of covarinace model. }

  \item{input.data}{The matrix \code{sim}. }

  \item{output.data}{The output matrix \eqn{h} in Details. }

  \item{time}{time (in seconds) required after preprocessing data in R, reported using,
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

set.seed(1)
n <- 1000
coords <- cbind(runif(n,0,1), runif(n,0,1))

sigma.sq = 1
phi = 5

set.seed(1)
sim <- matrix(rnorm(3*n),n, 3)
correlation_result <- BRISC_correlation(coords, sigma.sq = sigma.sq, phi = phi, sim = sim)
}

\keyword{model}
