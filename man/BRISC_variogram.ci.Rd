\name{BRISC_variogram.ci}
\alias{BRISC_variogram.ci}
\title{Function for plotting estimated Variogram and confidence region}

\description{
  The function \code{BRISC_variogram.ci} plots estimated Variogram and associated confience region.
  \code{BRISC_variogram.ci} uses the parameter estimates from \code{BRISC_estimation} and associated confidence interval from \code{BRISC_bootstrap}.
}

\usage{
BRISC_variogram.ci(BRISC_Out, confidence_est,
                   plot.variogram = FALSE)
}

\arguments{

  \item{BRISC_Out}{an object of class \code{BRISC_Out}, obtained as an output of \cr \code{BRISC_estimation}. }

  \item{confidence_est}{bootstrp sample of the Theta parameters, obtained from \code{BRISC_bootstrap}. }

  \item{plot.variogram}{if \code{TRUE}, plots the variogram and the associated confidence region. Default is \code{FALSE}. }

}

\value{
  A list comprising of the following:

  \item{variogram}{variogram and associated confidence region corresponding to lag ranging from 0 to 20, evaluated at 0.01 frequency. }

  \item{Plot}{plots the Variogram and associated confidence region with legends. }

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

estimation_result <- BRISC_estimation(coords, y, x)
bootstrap_result <- BRISC_bootstrap(estimation_result, n_boot = 10)
varg <- BRISC_variogram.ci(estimation_result,
                           bootstrap_result$boot.Theta,
                           plot.variogram = TRUE)

}

\keyword{model}
