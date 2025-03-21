### The code below is modified from code used in the paper 
### "Conditionally structured variational Gaussian approximation 
### with importance weights" by Linda S. L. Tan, Aishwarya Bhaskaran 
### and David J. Nott. (https://doi.org/10.1007/s11222-020-09944-8) 
### and kindly supplied by the authors of the paper.
data {
  int<lower=0> n; // number of subjects 
  int<lower=0> N; // number of obs 
  int<lower=0> k; // number of fixed effects
  int<lower=0,upper=1> y[N]; // responses
  matrix[N,k] X; // fixed effects covariates
  vector[N] Z; // random effects covariates 
  int<lower=1> startindex[n];
  int<lower=1> endindex[n];
}
parameters {
  vector[n] b;
  vector[k] beta;
  real zeta;
}
model {
  vector[N] prob;
  zeta ~ normal(0, 10);  // vectorized form, 10 is the standard deviation
  beta ~ normal(0, 10);  // vectorized form 
  b ~ normal(0,exp(-zeta));       // vectorized form
  for (i in 1:n) {
   for (j in startindex[i]:endindex[i]){
    prob[j] = dot_product(X[j,], beta) + Z[j]*b[i];
   }
  }
  y ~ bernoulli_logit(prob);
  //y ~ bernoulli(Phi(prob));
}
