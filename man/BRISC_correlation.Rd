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
BRISC_correlation(coords, sim, sigma.sq = 1, tau.sq = 0, phi = 1,
                  nu = 1.5, n.neighbors = NULL, n_omp = 1,
                  cov.model = "exponential",
                  search.type = "tree", stabilization = NULL,
                  verbose = TRUE, tol = 12)
}

\arguments{

  \item{coords}{an \eqn{n \times 2}{n x 2} matrix of the observation
  coordinates in \eqn{R^2} (e.g., easting and northing). }

  \item{sim}{an \eqn{n \times k}{n x k} matrix of the \eqn{k}{k} many \eqn{n \times 1}{n x 1} vectors
  from which the correlated data are calculated (see Details below). }

  \item{sigma.sq}{value of sigma square. Default value is 1. }

  \item{tau.sq}{value of tau square. Default value is 0.1. }

  \item{phi}{value of phi. Default value is 1. }

  \item{nu}{value of nu, only required for matern covariance model. Default value is 1.5. }

  \item{n.neighbors}{number of neighbors used in the NNGP. Default value is \eqn{max(100, n -1)}. We suggest a high value of
                     \code{n.neighbors} for lower value of phi. }

  \item{n_omp}{number of threads to be used, value can be more than 1 if source code is compiled with OpenMP support.
               Default is 1. }

  \item{cov.model}{keyword that specifies the covariance function to be used in modelling the spatial dependence structure
                   among the observations. Supported keywords are: \code{"exponential"}, \code{"matern"}, \code{"spherical"}, and \code{"gaussian"}
                   for exponential, Matern, spherical and Gaussian covariance function respectively. Default value is \code{"exponential"}. }

  \item{search.type}{keyword that specifies type of nearest neighbor search algorithm to be used. Supported keywords are:
                     \code{"brute"}, \code{"tree"} and \code{"cb"}. \cr \code{"brute"} and \code{"tree"} provide the same result, though
                     \code{"tree"} should be faster. \code{"cb"} implements fast code book search described in Ra and Kim (1993)
                     modified for NNGP. If locations do not have identical coordinate values on the axis used for the nearest neighbor
                     ordering (see \code{order} argument) then \code{"cb"} and \code{"brute"} should produce identical neighbor sets.
                     However, if there are identical coordinate values on the axis used for nearest neighbor ordering, then \code{"cb"}
                     and \code{"brute"} might produce different, but equally valid, neighbor sets, e.g., if data are on a grid.
                     Default value is \code{"tree"}. }

    \item{stabilization}{when we use a very smooth covarince model (lower values of phi for spherical and Gaussian
                      covariance and low phi and high nu for Matern covarinace) in absence of a non-negligble nugget, the correlation process may fail
                      due to computational instability. If \code{stabilization = TRUE}, performs stabilization by setting \code{tau.sq = } \eqn{max{\code{tau
                      .sq}, \code{sigma.sq} * 1e-06}}. Default value is \code{TRUE} for \code{cov.model = "expoenential"} and \code{FALSE} otherwise. }

  \item{verbose}{if \code{TRUE}, model specifications along with information regarding OpenMP support and progress of the algorithm
                 is printed to the screen. Otherwise, nothing is printed to the screen. Default value is \code{TRUE}. }

    \item{tol}{the input observation coordinates are rounded to this many places after the decimal. The default value is 12. }

}

\details{
Denote \eqn{g} be the input \code{sim}. Let \eqn{\Sigma} be the precision matrix associated with the covariance model determined by the \eqn{cov.model} and model parameters. Then \code{BRISC_correlation} calculates \eqn{h}, where \eqn{h} is given as follows:
\deqn{
S ^{-0.5} h = g
}
where, \eqn{S ^{-0.5}} is a sparse approximation of the cholesky factor \eqn{\Sigma ^{-0.5}} of the precision matrix \eqn{\Sigma ^{-1}}, obtained from NNGP.
}

\value{
  A list comprising of the following:

  \item{coords}{the matrix \code{coords}. }

  \item{n.neighbors}{the used value of \code{n.neighbors}. }

  \item{cov.model}{the used covariance model. }

  \item{Theta}{parameters of covarinace model; accounts for \code{stabilization}. }

  \item{input.data}{the matrix \code{sim}. }

  \item{output.data}{the output matrix \eqn{h} in Details. }

  \item{time}{time (in seconds) required after preprocessing data in R, \cr reported using,
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
phi = 1

set.seed(1)
sim <- matrix(rnorm(3*n),n, 3)
correlation_result <- BRISC_correlation(coords, sigma.sq = sigma.sq,
                                        phi = phi, sim = sim)
}

\keyword{model}
