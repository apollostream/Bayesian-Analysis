// "hierMD_MLT.stan" by Michael L. Thompson
// https://www.linkedin.com/in/mlthomps
// Nov. 22, 2020
//
// Adapted from the Stan code in R script "VB_stan_hierMD.R", 
// which included the following banner:
// ###    IPG - IDSIA
// ###    Authors: L. Azzimonti, G. Corani
// ###    Reference: "Hierarchical estimation of parameters in Bayesian networks",
// ###    Date: May 16, 2019
// ###
// ###    This code is for demostration purpose only
// ###    for licensing contact laura@idsia.ch
// ###    COPYRIGHT (C) 2019 IDSIA
data {
  int<lower=2> n_st_ch; // number of child states
  int<lower=2> n_st_pr; // number of total combos of parent states
  int<lower=0> N_ch_pr[n_st_ch,n_st_pr]; // number of cases at all combos of parents & child
  vector<lower=0>[n_st_ch] alpha_0; // hyperparameter for Dirichlet priors
}

parameters {
  simplex[n_st_ch] theta[n_st_pr]; // conditional probability table parameters
  simplex[n_st_ch] alpha_norm; // population-level parameter for Dirichlet priors, normalized
  real<lower=0> N_prior; // number of cases represented by prior
}

transformed parameters {
  vector<lower=0>[n_st_ch] alpha; // population-level parameter for Dirichlet priors
  alpha = N_prior * alpha_norm;
}

model {
  alpha_norm ~ dirichlet(alpha_0); // prior
  N_prior    ~ student_t(4,1,1);   // prior
  
  for (i_st_pr in 1:n_st_pr){
    theta[i_st_pr]    ~ dirichlet( alpha ); // prior
    N_ch_pr[,i_st_pr] ~ multinomial( theta[i_st_pr] ); // likelihood
  }
}
