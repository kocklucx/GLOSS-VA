### The code below is modified from code used in the paper 
### "Gaussian variational approximation with sparse precision
### matrices" by Linda S. L. Tan and David J. Nott.
### (https://doi.org/10.1007/s11222-017-9729-7) 
### and kindly supplied by the authors of the paper.

library(rstan)

####################
# Epilepsy dataset #
####################

df <-  read.table("data/data_epilepsy.txt", header=TRUE)
n <- dim(df)[1]                      # no. of subjects
vni <- rep(4,n)
X <- NULL
Z <- NULL
y <- NULL
lb4 <- log(df$Base4)
trt <- c(rep(0,28),rep(1,31))
lage <- log(df$Age)
clage <- scale(lage,scale=FALSE)
V4 <- c(0,0,0,1)
visit <- c(-3,-1,1,3)/10
lb4trt <- lb4*trt
N <- sum(vni)
startindex <- c(0, cumsum(vni)[1:(n-1)]) + 1
endindex <- cumsum(vni)

################
# random slope #
################
for (i in 1:n){
  y[[i]] <- c(df$Y1[i], df$Y2[i], df$Y3[i], df$Y4[i])
  Z[[i]] <- cbind(rep(1,vni[i]), visit)
  X[[i]] <- cbind(rep(1,vni[i]), rep(lb4[i],4), rep(trt[i],4), rep(lb4trt[i],4), rep(clage[i],4), visit)
}
sum(vni)
n <- length(y)
k <- dim(X[[1]])[2]
p <- dim(Z[[1]])[2]
d <- n*p + k + p*(p+1)/2
yall <- unlist(y)
Xall <- matrix(unlist(lapply(X,t)), ncol=k, byrow=TRUE)
Zall <- matrix(unlist(lapply(Z,t)), ncol=p, byrow=TRUE)
pzeros <- rep(0,p)
zetalength <- p*(p+1)/2
data <- list(n=n, N=as.integer(N), k=k, p=p, y=yall, X=Xall, Z=Zall, pzeros=pzeros,
             startindex=as.integer(startindex), endindex=as.integer(endindex), zetalength=as.integer(zetalength))


# stan (mcmc) #
start_time <- Sys.time()
fit <- stan(file = 'mcmc_sampling/stan_files/model_epilepsy_slope.stan',
            data = data, iter = 50000, chains = 1,thin=1)
end_time <- Sys.time()
print(end_time - start_time)

#
la <- extract(fit, permuted = TRUE, inc_warmup=FALSE) # return a list of arrays 
write.csv(la, "results/EpilepsyMCMC.csv", row.names = FALSE)

data_py <- as.data.frame(data$X)
colnames(data_py) <- c('intercept','base','treatment','base x treatment','age','visit')
data_py['y'] <- data$y
write.csv(data_py, "data/epilepsy.csv", row.names = FALSE)
