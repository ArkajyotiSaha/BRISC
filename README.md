# BRISC
Bootstrap for Rapid Inference on Spatial Covariances. The codes of package "liblbfgs (Naoaki Okazaki)" are also available here for user convenience. Some code snippets are borrowed from R package "spNNGP (Andrew Finley et al.)". We recommend compiling the code with OpenMP support, though the OpenMP flag is omitted for sake of user convenience.



In order to use the package follow these steps:

library(devtools) #If "devtools" is not installed, install "devtools" from CRAN before this step

install_github("ArkajyotiSaha/BRISC")

library(BRISC)


##For help on the function brisc please use the following:

?brisc
