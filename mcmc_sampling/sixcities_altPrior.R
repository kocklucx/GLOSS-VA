library(geepack)
library(rstan)

setwd("~/Desktop/skew_vb/revision1")

data(ohio)
age <- ohio$age
smoke <- ohio$smoke
smokeage <- ohio$smoke*ohio$age

ID <- unique(ohio$id)
n <- length(ID)                      # no. of subjects
vni <- rep(4,n)
X <- NULL
Z <- NULL
y <- NULL
for (i in 1:n){
  rows <- which(ohio$id == ID[i])
  vni[i] <- length(rows)
  y[[i]] <- ohio$resp[rows]
  Z[[i]] <- cbind(rep(1,vni[i]))
  X[[i]] <- cbind(rep(1,vni[i]), smoke[rows], age[rows], smokeage[rows])
}
n <- length(y)                              # no. of subjects
k <- dim(X[[1]])[2]                         # length of beta
p <- dim(Z[[1]])[2]                         # length of b_i 
d <- n*p + k + p*(p+1)/2
yall <- unlist(y)
Xall <- matrix(unlist(lapply(X,t)), ncol=k, byrow=TRUE)
Zall <- unlist(Z)
labels <- c('intercept','smoke','age','smoke x age','zeta')  
N <- sum(vni)
startindex <- c(0, cumsum(vni)[1:(n-1)]) + 1
endindex <- cumsum(vni)
data <- list(n=n, N=N, k=k, y=yall, X=Xall, Z=Zall, startindex=startindex, endindex=endindex)

########
# stan #
########
fit <- stan(file = "~/Desktop/skew_vb/revision1/mcmc_sampling/stan_files/sixcites_altPrior.stan",data = data, iter = 50000, chains = 1, thin=1)
la <- extract(fit, permuted = TRUE, inc_warmup=FALSE) # return a list of arrays 
write.csv(la, "results/sixcities_altPriorMCMC.csv", row.names = FALSE)
write.csv(ohio,"data/ohio.csv")
