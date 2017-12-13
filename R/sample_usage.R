library(devtools) #If "devtools" is not installed, install "devtools" from CRAN before this step

install_github("ArkajyotiSaha/BRISC")

library(BRISC)

rmvn <- function(n, mu = 0, V = matrix(1)){
    p <- length(mu)
    if(any(is.na(match(dim(V),p))))
        stop("Dimension not right!")
    D <- chol(V)
    t(matrix(rnorm(n*p), ncol=p)%*%D + rep(mu,rep(n,p)))
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

y <- rnorm(n, x%*%B + w, sqrt(tau.sq))

ptm1 <- proc.time()
result <- brisc(coords, x, y, n_boot = 10, n_omp = 1)
proc.time() - ptm1
result$estimated.theta ## Gives estimation of (sigma.sq, tau.sq, phi)
result$estimated.beta ## Gives estimation of (Beta)
result$confidence.interval ## Gives estimation of (Beta)
