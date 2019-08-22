BRISC_variogram.ci <- function(BRISC_Out, confidence_est, plot.variogram = FALSE){
  cov.model <- BRISC_Out$cov.model

  if(cov.model=="exponential"){
    variog <- function(t, est_theta){
      return(est_theta[1] * (1 - exp(-t*est_theta[3]))+est_theta[2])
    }
  }
  if(cov.model=="gaussian"){
    variog <- function(t, est_theta){
      return(est_theta[1] * (1 - exp(-(t^2)*(est_theta[3]^2)))+est_theta[2])
    }
  }
  if(cov.model=="spherical"){
    variog <- function(t, est_theta){
      if(t*est_theta[3] > 1){
        resp <- est_theta[1]
      }else{
        resp <- est_theta[1] * (t*est_theta[3] * 3/2 - 0.5*(t*est_theta[3])^3)
      }
      return(resp+est_theta[2])
    }
  }
  if(cov.model=="matern"){stop("error: NA for matern covariance model\n")}
  x_sept <- seq(0,20, by = 0.01)
  estimate_new <- sapply(x_sept, variog, BRISC_Out$Theta)

  res_loc_boot <- matrix(0, nrow(confidence_est), length(x_sept))

  variog_ci_calc <- function(i, x_sept, confidence_est, estimate_new, res_loc_boot){
    for(inti in 1:nrow(confidence_est)){
      res_loc_boot[inti,i] <- variog(x_sept[i], confidence_est[inti,])
    }
    c(2*estimate_new[i] - quantile(res_loc_boot[,i], c(.975,.025)),estimate_new[i], x_sept[i])
  }

  variog_plot <- t(sapply(1:length(x_sept), variog_ci_calc, x_sept, confidence_est, estimate_new, res_loc_boot))
  variog_plot <- variog_plot[,c(4, 1, 3, 2)]

  colnames(variog_plot) <- c("x", "2.5%", "Estimate", "97.5%")
  result_list <- list()
  result_list$variogram <- variog_plot
  result_list
  if(plot.variogram){
    upper_lim <- max(variog_plot[,4])*1.1+50
    lower_lim <- min(variog_plot[,2])*0.9
    plot(variog_plot[,1], variog_plot[,2], type="l", col="red", xlab="Lag",
         ylab="Gamma", ylim = c(lower_lim, upper_lim), lwd = 2, lty = 1)
    lines(variog_plot[,1], variog_plot[,3], col="blue", type="l", lwd = 2, lty = 2)
    lines(variog_plot[,1], variog_plot[,4],  col="black", type="l", lwd = 2, lty = 3)
    legend("topright" , legend=c(colnames(variog_plot)[-1]),
           col=c("red", "blue", "black"), cex=0.8, lty = 1:3)
  }
}
