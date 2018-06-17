\name{BRISC_estimation}
\alias{BRISC_estimation}
\title{Function for estimation with BRISC}

\description{
  The function \code{BRISC_estimation} fits univariate spatial regression models for large
  spatial data using Vecchia's approximate likelihood (Vecchia, 1988). \code{BRISC_estimation}
  uses the sparse Cholesky representation of Vecchia’s likelihood developed in Datta et al., 2016.
  The Maximum Likelihood Estimates (MLE) of the parameters are used later for calculating the
  confidence interval via the \code{BRISC_bootstrap} (BRISC, Saha & Datta, 2018). We recommend
  using \code{BRISC_estimation} followed by \code{BRISC_bootstrap} to obtain the cofidence
  intervals for the model parameters.

  The optimization is performed with C library of limited-memory BFGS
  libLBFGS: a library of Limited-memory Broyden-Fletcher-Goldfarb-Shanno (L-BFGS),
  http://www.chokkan.org/software/liblbfgs/ (Naoaki Okazaki). For user convenience
  the soure codes of the package libLBFGS are provided in the package. The code for
  the coordinate ordering method, approximate Maximum Minimum Distance (Guinness, 2018)
  is available in https://github.com/joeguinness/gp_reorder/tree/master/R and is adopted
  with minor modification. Some code blocks are borrowed from the R package: spNNGP:
  Spatial Regression Models for Large Datasets using Nearest Neighbor Gaussian Processes
  https://CRAN.R-project.org/package=spNNGP .

}

\usage{
BRISC_estimation(coords, x, y, sigma.sq = 1, tau.sq = 0.1, phi = 1,
                 nu = 0.5, n.neighbors = 15, n_omp = 1, order = "AMMD",
                 cov.model = "exponential", search.type = "tree",
                 verbose = TRUE, eps = 2e-05)
}

\arguments{

  \item{coords}{an \eqn{n \times 2}{n x 2} matrix of the observation
  coordinates in \eqn{R^2}{R^2} (e.g., easting and northing). }

  \item{x}{an \eqn{n \times p}{n x p} matrix of the covariates in the observation coordinates. }

  \item{y}{a \eqn{n}{n} length vector of response at the observed coordinates. }

  \item{sigma.sq}{starting value of sigma square. Default value is 1. }

  \item{tau.sq}{starting value of tau square. Default value is 0.1. }

  \item{phi}{starting value of phi. Default value is 1. }

  \item{nu}{starting value of nu, only required for matern covariance model. Default value is 0.5. }

  \item{n.neighbors}{number of neighbors used in the NNGP. Default value is 15. }

  \item{n_omp}{number of threads to be used, value can be more than 1 if source code is compiled with OpenMP support.
               Default is 1. }

  \item{order}{Keyword that specifies the ordering scheme to be used in ordering the observations. Supported keywords are:
               "AMMD" and "Sum_coords" for approximate Maximum Minimum Distance and sum of coordinate based ordering,
               respectively. Default value is "AMMD". }

  \item{cov.model}{Keyword that specifies the covariance function to be used in modelling the spatial dependence structure
                   among the observations. Supported keywords are: "exponential", "matern", "spherical", and "gaussian"
                   for exponential, matern, spherical and gaussian covariance function respectively. Default value is "exponential". }

  \item{search.type}{Keyword that specifies type of nearest neighbor search algorithm to be used. Supported keywords are:
                     "tree" and "brute". Both of them provide the same result, though "tree" should be faster.
                     Default value is "tree". }

  \item{verbose}{if TRUE, model specifications along with information regarding OpenMP support and progress of the algorithm
                 is printed to the screen. Otherwise, nothing is printed to the screen. Default value is FALSE. }

  \item{eps}{The tolerance to be used in centred finite difference approximation of derivatives. Default value is 2e-05. }
}

\value{
  An object of class \code{BRISC_Out}, which is a list comprising:

  \item{ord}{the vector of indices used to order data necessary for fitting the NNGP
    model.}

  \item{coords}{the matrix \code{coords[ord,]}. }

  \item{y}{the vector \code{y[ord]}. }

  \item{X}{the matrix \code{x[ord,,drop=FALSE]}. }

  \item{n.neighbors}{the used value of \code{n.neighbors}. }

  \item{cov.model}{the used covariance model. }

  \item{eps}{value of used \code{eps}. }

  \item{init}{Initial values of the parameters of the covariance model. }

  \item{Beta}{estimate of beta. }

  \item{Theta}{estimate of parameters of covarinace model. }

   \item{estimation.time}{time (in seconds) required to perform the model fitting after ordering and preprocessing data in R,
                          reported using \code{proc.time()}. }

  \item{BRISC_Object}{Object required for bootstrap and prediction. }

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
  DOI: 10.1080/00401706.2018.1437476, https://github.com/joeguinness/gp_reorder/tree/master/R .

  Vecchia, A. V. (1988) Estimation and model identification for
  continuous spatial processes. Journal of the Royal Statistical
  Society. Series B (Methodological), 297-312.

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

estimation_result <- BRISC_estimation(coords, x, y)
estimation_result$Theta ## Gives estimation of covariance model parameters.
estimation_result$Beta ## Gives estimation of Beta

}

\keyword{model}
