# BRISC
Bootstrap for Rapid Inference on Spatial Covariances: Provides parameter estimates and bootstrap based confidence intervals for all parameters in a Gaussian Process based spatial regression model.


In order to use the package follow these steps:

library(devtools) #If "devtools" is not installed, install "devtools" from CRAN before this step

install_github("ArkajyotiSaha/BRISC")

library(BRISC)


##For help on the functions in Brisc please use the following:

?BRISC_estimation #(for estimation)
?BRISC_bootstrap #(for bootstrap)
?BRISC_prediction #(for prediction)



PS: The code of package "liblbfgs (Naoaki Okazaki)" are also available here for user convenience. Some code snippets are borrowed from R package "spNNGP (Andrew Finley et al.)". The code for approximate MMD ordering is borrowed from https://github.com/joeguinness/gp_reorder after minor modifications. We recommend compiling the code with OpenMP support, though the OpenMP flag is omitted for sake of user convenience. The code for covariance models other than exponential model are in beta testing stage.



# References:
1. Vecchia, A. V. (1988). Estimation and model identification for continuous spatial processes. Journal of the Royal Statistical Society. Series B (Methodological), 297-312.

2. Datta, A., Banerjee, S., Finley, A. O., & Gelfand, A. E. (2016). Hierarchical nearest-neighbor Gaussian process models for large geostatistical datasets. Journal of the American Statistical Association, 111(514), 800-812.

3. Guinness, J. (2018). Permutation and Grouping Methods for Sharpening Gaussian Process Approximations. Technometrics, (just-accepted).

