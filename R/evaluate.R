arrange<-function(x_r){
  c_matrix=matrix(0,length(x_r),length(x_r[[1]]))
  for(i in 1:length(x_r)){
    c_matrix[i,]<-(x_r[[i]])
  }
  c_matrix
}

