# Variational inference for hierarchical models with conditional scale and skewness corrections
Code for the paper "Variational inference for hierarchical models with conditional scale and skewness corrections" by Lucas Kock, Linda S. L. Tan, Prateek Bansal and David J. Nott.

Gaussian variational approximations are widely used for summarizing posterior distributions in Bayesian models, especially in high-dimensional settings. However, a drawback of such approximations is the inability to capture skewness or more complex features of the posterior. Recent work suggests applying skewness corrections to Gaussian or other symmetric approximations to address this limitation. We propose to incorporate the skewness correction into the definition of an approximating variational family. We consider approximating the posterior for hierarchical models, in which there are global and local parameters.  A baseline variational approximation is first defined as the product of a Gaussian marginal posterior for global parameters and a Gaussian conditional posterior for local parameters given the global ones.  Skewness corrections are then considered. The adjustment of the conditional posterior term for local variables is adaptive to the global parameter value.    
Optimization of baseline variational parameters is performed jointly with the skewness correction. Our approach allows the location, scale and skewness to be captured separately, without using additional parameters for skewness adjustments.  The proposed method substantially improves accuracy for only a modest increase in computational cost compared to state-of-the-art Gaussian approximations. Good performance is demonstrated in generalized linear mixed models and multinomial logit discrete choice models. 

## Reproduction of results from the paper
To reproduce the experiments from the paper you need to run the MCMC sampling first and then the respective .py file. 

  a) sixcities reproduces the logistic mixed model example
  
  b) epilepsy reproduces the Poisson mixed model example
  
  c) parking reproduces the discrete choice model example
  
  d) elbo_plots reproduces the "Estimated ELBO versus iteration number" plot as shown in the paper

## Apply GLOSS-VA to your own models
The file approximation.py contains code to approximate hierarchical models with G-VA, CSG-VA, and GLOSS-VA. More comments on the code can be found within this file. In particular:

  a) Variational_Approximation(...,skewness=False,variance=False) is a Gaussian variational approximation (G-VA)
  
  b) Variational_Approximation(...,skewness=False,variance=True) is a conditional structured Gaussian approximation (CSG-VA)
  
  c) Variational_Approximation(...,skewness=True,variance=True) is our novel variational approximation (GLOSS-VA)
  
epilepsy.py is a simple example to understand how the method can be applied to new model specifications. 
