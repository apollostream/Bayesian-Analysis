// "hierMDmix_MLT.stan" by Michael L. Thompson
// https://www.linkedin.com/in/mlthomps
// Nov. 23, 2020
// Inspired by hierarchical Multinomial Dirichlet approach of Azzimonti & Corani
// in their R script "VB_stan_hierMD.R", which included the following banner:
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
  
  int <lower=1> n_parent; // number of parents
  int <lower=2> n_st_pr_i[n_parent]; // number of states for each parent
  int i_st_pr[n_st_pr,n_parent]; // state of each parent for each combo of parents
      // in R: > i_st_pr <- as.matrix(expand.grid(purrr::map(n_st_pr_i,~seq(1,.x))))
}

transformed data {
  int n_st_pr_sum; // sum of number of states for parents
  vector[n_parent] alpha_pr_mix; // Dirichlet parameters for mixture
  
  n_st_pr_sum  = sum(n_st_pr_i);
  alpha_pr_mix = rep_vector(1,n_parent);
}

parameters {
  simplex[n_st_ch] theta[n_st_pr];  // conditional probability table parameters
  simplex[n_st_ch] alpha_norm; // population-level parameter for Dirichlet priors, normalized
  simplex[n_st_ch] alpha_i_norm[n_st_pr_sum];// prior for the local states
  real<lower=0> N_prior;  // number of cases represented by prior
  simplex[n_parent] p_mix; // mixture probabilities on the parents alpha_i
}

transformed parameters {
  vector<lower=0>[n_st_ch] alpha_ch[n_st_pr]; // mixture alpha
  vector<lower=0>[n_st_ch] alpha; // population level
  vector<lower=0>[n_st_ch] alpha_i[n_st_pr_sum];// prior for local states
  
  // unnormalized hyperparameters
  alpha = N_prior * alpha_norm;  // population-level
  for(i in 1:n_st_pr_sum){
    alpha_i[i] = N_prior * alpha_i_norm[i]; // parent state-level
  }

  // Mix alpha hyperparameter for the prior over each parent's state
  for( i_st in 1:n_st_pr ){
    alpha_ch[i_st] = rep_vector(0,n_st_ch); // initialize as zeros
    for( i in 1:n_parent ){
      // cummulative mixture contributions of parent states
      alpha_ch[i_st] += p_mix[i] * alpha_i[ sum(head(n_st_pr_i,i-1)) + i_st_pr[i_st,i] ];
    }
  }
}

model {
  
  alpha_norm ~ dirichlet( alpha_0 );      // prior
  N_prior    ~ student_t( 4, 1, 1 );      // prior
  p_mix      ~ dirichlet( alpha_pr_mix ); // prior
  
  // sample alpha hyperparameter for each state of each parent
  {
    int i_st = 0;
    for( i in 1:n_parent ){
      for( j in 1:n_st_pr_i[i]){
        i_st += 1;
        alpha_i_norm[i_st] ~ dirichlet( alpha ); // prior
      }
    }
  }

  for (i_st in 1:n_st_pr){
    theta[i_st]    ~ dirichlet( alpha_ch[i_st] ); // prior
    N_ch_pr[,i_st] ~ multinomial( theta[i_st] );  // likelihood
  }
}
