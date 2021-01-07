\name{BRISC_simulation}
\alias{BRISC_simulation}
\title{Function to simulate data with BRISC}

\description{
  The function \code{BRISC_simulation} simulates correlated data (known structure) using Nearest Neighbor
  Gaussian Processes (NNGP). \code{BRISC_simulation} uses the sparse Cholesky representation of Vecchia’s
  likelihood developed in Datta et al., 2016. \code{BRISC_simulation} uses \code{\link{BRISC_correlation}}
  for this purpose.
}

\usage{
BRISC_simulation(coords, sim_number = 1,
                 seeds =  NULL, sigma.sq = 1,
                 tau.sq = 0, phi = 1, nu = 1.5,
                 n.neighbors = NULL, n_omp = 1,
                 cov.model = "exponential",
                 search.type = "tree",
                 stabilization = NULL,
                 verbose = TRUE, tol = 12)
}

\arguments{

  \item{coords}{an \eqn{n \times 2}{n x 2} matrix of the observation
  coordinates in \eqn{R^2} (e.g., easting and northing). }

  \item{sim_number}{number of simulations. Default value is 1. }

  \item{seeds}{seeds which are used in generation of the initial independent data. Default value is \code{NULL}.
  If non-null, the number of seeds must be equal to \code{sim_number}. }

  \item{sigma.sq}{value of sigma square. Default value is 1. }

  \item{tau.sq}{value of tau square. Default value is 0.1. }

  \item{phi}{value of phi. Default value is 1. }

  \item{nu}{starting value of nu, only required for matern covariance model. Default value is 1.5. }

  \item{n.neighbors}{number of neighbors used in the NNGP. Default value is 15. }

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

\value{
 A list comprising of the following:

  \item{coords}{the matrix \code{coords}. }

  \item{n.neighbors}{the used value of \code{n.neighbors}. }

  \item{cov.model}{the used covariance model. }

  \item{Theta}{parameters of covarinace model; accounts for \code{stabilization}. }

  \item{input.data}{the \eqn{n \times sim_number} matrix of generated independent data. Here \eqn{i^{th}} column
  denotes the data corresponding to the \eqn{i^{th}} simulation.  }

  \item{output.data}{the \eqn{n \times sim_number} matrix of generated correlated data. Here \eqn{i^{th}} column
  denotes the data corresponding to the \eqn{i^{th}} simulation.  }

  \item{time}{time (in seconds) required after preprocessing data in \code{R}, \cr reported using,
                          \code{proc.time()}. }
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

simulation_result <- BRISC_simulation(coords, sim_number = 3)

}

\keyword{model}
