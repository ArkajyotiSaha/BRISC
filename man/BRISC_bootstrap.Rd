\name{BRISC_bootstrap}
\alias{BRISC_bootstrap}
\title{Function for performing bootstrap with BRISC}

\description{
  The function \code{BRISC_bootstrap} performs bootstrap on univariate spatial
  regression models using outputs of \code{BRISC_estimation}.
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

estimation_result <- BRISC_estimation(coords, x, y)
bootstrap_result <- BRISC_bootstrap(estimation_result, n_boot = 10)

}

\keyword{model}
