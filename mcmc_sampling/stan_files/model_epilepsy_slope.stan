### The code below is modified from code used in the paper 
### "Gaussian variational approximation with sparse precision
### matrices" by Linda S. L. Tan and David J. Nott.
### (https://doi.org/10.1007/s11222-017-9729-7) 
### and kindly supplied by the authors of the paper.
data {
  int<lower=0> n; // number of subjects 
  int<lower=0> N; // number of obs 
  int<lower=0> k; // number of fixed effects
  int<lower=0> p; // number of random effects
  int<lower=0> zetalength; // 
  int<lower=0> y[N]; // responses
  matrix[N,k] X; // fixed effects covariates
  matrix[N,p] Z; // random effects covariates 
  int<lower=1> startindex[n];
  int<lower=1> endindex[n];
  vector[p] pzeros;
}
parameters {
  matrix[p,n] b;
  vector[k] beta;
  vector[zetalength] zeta;
}
model {
  vector[N] prob;
  matrix[p,p] W;
  zeta ~ normal(0, 10);  // vectorized form, 10 is the standard deviation
  beta ~ normal(0, 10);  // vectorized form  
  W[1,1] = exp(zeta[1]);
  W[1,2] = 0;
  W[2,1] = zeta[2];
  W[2,2] = exp(zeta[3]);
  for (i in 1:n) {
   b[,i] ~ multi_normal(pzeros, W*W'); 
   for (j in startindex[i]:endindex[i]){
    prob[j] = dot_product(X[j,], beta) + dot_product(Z[j,], b[,i]);
   }
  }
  y ~ poisson_log(prob);
}
