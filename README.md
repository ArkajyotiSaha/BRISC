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



PS: The codes of package "liblbfgs (Naoaki Okazaki)" are also available here for user convenience. Some code snippets are borrowed from R package "spNNGP (Andrew Finley et al.)". The code for approximate MMD ordering is borrowed from https://github.com/joeguinness/gp_reorder after minor modifications. We recommend compiling the code with OpenMP support, though the OpenMP flag is omitted for sake of user convenience.
