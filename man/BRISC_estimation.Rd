\name{BRISC_estimation}
\alias{BRISC_estimation}
\title{Function for estimation with BRISC}

\description{
  The function \code{BRISC_estimation} fits univariate spatial regression models for large
  spatial data using Vecchia's approximate likelihood (Vecchia, 1988). \code{BRISC_estimation}
  uses the sparse Cholesky representation of Vecchia’s likelihood developed in Datta et al., 2016.
  The Maximum Likelihood Estimates (MLE) of the parameters are used later for calculating the
  confidence interval via the \code{BRISC_bootstrap} (BRISC, Saha & Datta, 2018). \cr We recommend
  using \code{BRISC_estimation} followed by \code{BRISC_bootstrap} to obtain the cofidence
  intervals for the model parameters.

  The optimization is performed with C library of limited-memory BFGS
  libLBFGS: a library of Limited-memory Broyden-Fletcher-Goldfarb-Shanno (L-BFGS), \cr
  http://www.chokkan.org/software/liblbfgs/ (Naoaki Okazaki). For user convenience
  the soure codes of the package libLBFGS are provided in the package. The code for
  the coordinate ordering method, approximate Maximum Minimum Distance (Guinness, 2018)
  is available in \cr https://github.com/joeguinness/gp_reorder/tree/master/R and is adopted
  with minor modification. Some code blocks are borrowed from the R package: spNNGP:
  Spatial Regression Models for Large Datasets using Nearest Neighbor Gaussian Processes
  https://CRAN.R-project.org/package=spNNGP .

}

\usage{
BRISC_estimation(coords, y, x = NULL, sigma.sq = 1,
                 tau.sq = 0.1, phi = 1,
                 nu = 1.5, n.neighbors = 15,
                 n_omp = 1, order = "Sum_coords",
                 cov.model = "exponential",
                 search.type = "tree",
                 stabilization = NULL,
                 pred.stabilization = 1e-8,
                 verbose = TRUE, eps = 2e-05,
                 nugget_status = 1, tol = 12)
}

\arguments{

  \item{coords}{an \eqn{n \times 2}{n x 2} matrix of the observation
  coordinates in \eqn{R^2}{R^2} (e.g., easting and northing). }

  \item{y}{an \eqn{n}{n} length vector of response at the observed coordinates. }

  \item{x}{an \eqn{n \times p}{n x p} matrix of the covariates in the observation coordinates. Default value is
  \eqn{n \times 1}{n x 1} matrix of \eqn{1} to adjust for the mean(intercept). }

  \item{sigma.sq}{starting value of sigma square. Default value is 1. }

  \item{tau.sq}{starting value of tau square. Default value is 0.1. }

  \item{phi}{starting value of phi. Default value is 1. }

  \item{nu}{starting value of nu, only required for matern covariance model. Default value is 1.5. }

  \item{n.neighbors}{number of neighbors used in the NNGP. Default value is 15. }

  \item{n_omp}{number of threads to be used, value can be more than 1 if source code is compiled with OpenMP support.
               Default is 1. }

  \item{order}{keyword that specifies the ordering scheme to be used in ordering the observations. Supported keywords are:
               \code{"AMMD"} and \code{"Sum_coords"} for approximate Maximum Minimum Distance and sum of coordinate based ordering,
               respectively. Default value is \code{"Sum_coords"}. \eqn{n > 65} is required for \code{"AMMD"}.}

  \item{cov.model}{keyword that specifies the covariance function to be used in modelling the spatial dependence structure
                   among the observations. Supported keywords are: \code{"exponential"}, \code{"matern"}, \code{"spherical"}, and \code{"gaussian"}
                   for exponential, Matern, spherical and Gaussian covariance function respectively. Default value is "exponential". }

  \item{search.type}{keyword that specifies type of nearest neighbor search algorithm to be used. Supported keywords are:
                     \code{"brute"}, \code{"tree"} and \code{"cb"}. \cr \code{"brute"} and \code{"tree"} provide the same result, though
                     \code{"tree"} should be faster. \code{"cb"} implements fast code book search described in Ra and Kim (1993)
                     modified for NNGP. If locations do not have identical coordinate values on the axis used for the nearest neighbor
                     ordering (see \code{order} argument) then \code{"cb"} and \code{"brute"} should produce identical neighbor sets.
                     However, if there are identical coordinate values on the axis used for nearest neighbor ordering, then \code{"cb"}
                     and \code{"brute"} might produce different, but equally valid, neighbor sets, e.g., if data are on a grid.
                     Default value is \code{"tree"}. }

  \item{stabilization}{when the spatial errors are generated from a very smooth covarince model (lower values of phi for spherical and Gaussian
                      covariance and low phi and high nu for Matern covarinace), the estimation process may fail due to computational
                      instability. If \code{stabilization = TRUE}, performs stabilization by adding a white noise to the reordered data with
                      nugget \code{tau.sq = } \code{sigma.sq} * 1e-06. Estimation is performed on this new data with \code{nugget_status = 1}
                      (see \code{nugget_status} argument below). Default value is \code{TRUE} for \code{cov.model = "expoenential"} and \code{FALSE} otherwise. }

  \item{pred.stabilization}{if not \code{NULL}, will truncate the estimated tau square to \code{pred.stabilization} * estimated sigma square.
                            This provides additional stability in \cr \code{BRISC_prediction}. Default value is \eqn{1e-8}. }

  \item{verbose}{if \code{TRUE}, model specifications along with information regarding OpenMP support and progress of the algorithm
                 is printed to the screen. Otherwise, nothing is printed to the screen. Default value is \code{TRUE}. }

  \item{eps}{tolerance to be used in centred finite difference approximation of derivatives. Default value is 2e-05. }

  \item{nugget_status}{if \code{nugget_status = 0}, \code{tau.sq} is fixed to 0, if \code{nugget_status = 1} \code{tau.sq} is estimated. Default value is 1. }

  \item{tol}{the input observation coordinates, response and the covariates are rounded to this many places after the decimal. The default value is 12. }
}

\value{
  An object of class \code{BRISC_Out}, which is a list comprising:

  \item{ord}{the vector of indices used to order data necessary for fitting the NNGP
    model.}

  \item{coords}{the matrix \code{coords[ord,]}. }

  \item{y}{If \code{stabilization = FALSE}, returns the vector \code{y[ord]}. \cr If \code{stabilization = TRUE}, returns \code{y[ord]} + used white noise in stabilization process. }

  \item{X}{the matrix \code{x[ord,,drop=FALSE]}. }

  \item{n.neighbors}{the used value of \code{n.neighbors}. }

  \item{cov.model}{the used covariance model. }

  \item{eps}{value of used \code{eps} for approximate derivation. Default value is 2e-05. }

  \item{init}{initial values of the parameters of the covariance model; \cr accounts for \code{stabilization}. }

  \item{Beta}{estimate of beta. }

  \item{Theta}{estimate of parameters of covarinace model. }

   \item{estimation.time}{time (in seconds) required to perform the model fitting after ordering and preprocessing data in \code{R},
                          reported using \code{proc.time()}. }

  \item{BRISC_Object}{object required for bootstrap and prediction. }

}

\references{
  Saha, A., & Datta, A. (2018). BRISC: bootstrap for rapid inference on spatial
  covariances. Stat, e184, DOI: 10.1002/sta4.184.

  Datta, A., S. Banerjee, A.O. Finley, and A.E. Gelfand. (2016)
  Hierarchical Nearest-Neighbor Gaussian process models for large
  geostatistical datasets. Journal of the American Statistical
  Association, 111:800-812.

  Guinness, J. (2018) Permutation and Grouping Methods for
  Sharpening Gaussian Process Approximations, Technometrics,
  DOI: 10.1080/00401706.2018.1437476, \cr
  https://github.com/joeguinness/gp_reorder/tree/master/R .

  Vecchia, A. V. (1988) Estimation and model identification for
  continuous spatial processes. Journal of the Royal Statistical
  Society. Series B (Methodological), 297-312.

  Okazaki N. libLBFGS: a library of Limited-memory Broyden-Fletcher-Goldfarb-Shanno
  (L-BFGS), \cr http://www.chokkan.org/software/liblbfgs/ .

  Andrew Finley, Abhirup Datta and Sudipto Banerjee (2020). spNNGP: Spatial Regression
  Models for Large Datasets using Nearest Neighbor Gaussian Processes. R package version 0.1.4.
  https://CRAN.R-project.org/package=spNNGP

  Ra, S. W., & Kim, J. K. (1993). A fast mean-distance-ordered partial codebook search algorithm
  for image vector quantization. IEEE Transactions on Circuits and Systems II: Analog and Digital
  Signal Processing, 40(9), 576-579.

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
phi = 1
tau.sq = 0.1

B <- as.matrix(beta)
D <- as.matrix(dist(coords))
R <- exp(-phi*D)
w <- rmvn(1, rep(0,n), sigma.sq*R)

y <- rnorm(n, x\%*\%B + w, sqrt(tau.sq))

estimation_result <- BRISC_estimation(coords, y, x)
estimation_result$Theta ##Estimates of covariance model parameters.
estimation_result$Beta ##Estimates of Beta

}

\keyword{model}
