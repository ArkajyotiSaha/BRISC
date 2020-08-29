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
BRISC_simulation(coords, sim_number = 1, seeds =  NULL, sigma.sq = 1, tau.sq = 0, phi = 1, nu = 0.5, n.neighbors = 15,
                  n_omp = 1, cov.model = "exponential", search.type = "tree", verbose = TRUE)
}

\arguments{

  \item{coords}{an \eqn{n \times 2}{n x 2} matrix of the observation
  coordinates in \eqn{R^2} (e.g., easting and northing). }

  \item{sim_number}{The number of simulations. Default value is 1. }

  \item{seeds}{The seeds which are used in generation of the initial independent data. Default value is \code{NULL}.
  If non-null, the number of seeds must be equal to \code{sim_number}. }

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

\value{
 A list comprising of the following:

  \item{coords}{the matrix \code{coords}. }

  \item{n.neighbors}{the used value of \code{n.neighbors}. }

  \item{cov.model}{the used covariance model. }

  \item{Theta}{Parameters of covarinace model. }

  \item{input.data}{The \eqn{n \times sim_number} matrix of generated independent data. Here \eqn{i^{th}} column
  denotes the data corresponding to the \eqn{i^{th}} simulation.  }

  \item{output.data}{The \eqn{n \times sim_number} matrix of generated correlated data. Here \eqn{i^{th}} column
  denotes the data corresponding to the \eqn{i^{th}} simulation.  }

  \item{time}{time (in seconds) required after preprocessing data in R, reported using,
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
phi = 5

simulation_result <- BRISC_simulation(coords, sigma.sq = sigma.sq, phi = phi, sim_number = 3, seeds = c(1,2,3))

}

\keyword{model}
