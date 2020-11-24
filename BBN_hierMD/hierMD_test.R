# Michael L. Thompson, Nov. 23, 2020
# Test Hierarchical Multinomial-Dirichlet ("hierMD") priors for Conditional
# Probability Tables (CPTs) of Bayesian Networks. 
# 1. The base case proposed by L. Azzimonti, G. Corani in: "Hierarchical
#    estimation of parameters in Bayesian networks",
# 2. A mixture model of parent states proposed by Thompson (me).
# 

# Conclusion (w/out having done any other testing than what's shown below):
# The more complicated mixture model gives results about the same as that of
# the Azzimonti-Corani model -- seem to be the same within the uncertainty of
# the parameter posteriors.
# Both are an improvement over a flat prior -- "BDeu" (Laplace smoothing).
#

# PACKAGE LOADING ====
library(magrittr) # I always use piping!
library(tidyverse) # Thank heaven & Hadly for the Tidyverse!

library(bnlearn)
library(gRain)

library(rstan)
# library(MCMCpack)
# library(coda)

select <- dplyr::select


# DATA LOADING ====
# Use the Movie Recommender System built earlier.
# Get a BBN to experiment with.
load("Rec_Sys.RData",verbose=TRUE)
bbn_info <- rec_sys$bn_models$`Star Trek: The Motion Picture (1979)`

# plot it
g1 <- graphviz.plot( bbn_info$tanb, render = FALSE )
g1 %>% 
  plot( 
    attrs = list(node=list(fontsize="36")), 
    main = bbn_info$target_movie_title
  )
df  <- as_tibble( bbn_info$bn_df2 )
bbn <- bbn_info$tanb_grn

# Augment with a "Has_Seen" node:
movie <- "Str.ctr_79"
lvls  <- bbn$universe$levels[[movie]]
has_seen_node_cpt <- cptable( 
  c("Has_Seen",movie),
  levels=c( list(Has_Seen=c("yes","no")), lvls ),
  values = matrix(
    c(ifelse(grepl("^r",lvls),1,0), ifelse(grepl("unseen",lvls),1,0) ),
    nrow  = 2, 
    byrow = TRUE
  )
)
bbn <- grain( compileCPT( c(list(has_seen_node_cpt), bbn$cptlist) ) )

# It's a ratings-based network, so augment with the "LIKE" node
like_node_cpt <- cptable( 
  c("LIKE",movie),
  levels = list(c("yes","no","unseen"),lvls) %>% set_names(c("LIKE",movie)),
  values = matrix(
    c(
      ifelse(grepl("^(r[1-4]=)|unseen",lvls),0,1),
      ifelse(grepl("^r[1-4]=",lvls),1,0),
      ifelse(grepl("unseen",lvls),1,0)
    ),
    nrow  = 3, 
    byrow = TRUE
  )
)
bbn <- grain( compileCPT( c(list(like_node_cpt), bbn$cptlist) ) )

querygrain(bbn,"Has_Seen")
querygrain(bbn,"LIKE")
querygrain(bbn,"LIKE", evidence = c(Has_Seen="yes") )

# STAN COMPILATION ====
# Compile the Stan programs of the two model variants.
sm_hierMD    <- stan_model(file="hierMD_MLT.stan",    model_name="hierMD")
sm_hierMDmix <- stan_model(file="hierMDmix_MLT.stan", model_name="hierMDmix")

# CPT ESTIMATION ====
gen_CPT_hierMD <- function(bbn, df, ch_name = "gender", full = FALSE){
  # Let's first do the "gender" node. It has 2 parents: "Str.ctr_79" and "occupation".
  #ch_name <- "gender"
  cpt <- bbn$cptlist[[ch_name]] # array with CPT using BDeu priors (Laplace smoothing.)
  # Build data list for Stan program sm_hierMD
  n_st_ch  <- dim(cpt)[[1]]
  pr_names <- setdiff( names(dim(cpt)), ch_name )
  df_ch_pr <- df %>%
    count(across(all_of(c(rev(pr_names),ch_name))),.drop = FALSE) %>%
    unite(col = "pr_combo", all_of(pr_names), sep = "#")
  N_ch_pr <- matrix(df_ch_pr$n, nrow = n_st_ch)
  
  theta_BDeu1    <- t(N_ch_pr + (1/n_st_ch)) %>% divide_by(rowSums(.))
  # These should match "cpt"
  theta_BDeu_eps <- t(N_ch_pr + (1/length(N_ch_pr))) %>% divide_by(rowSums(.))
  # spot-check...
  #(cpt[,,1,drop=FALSE]) %>% round(3)
  #theta_BDeu_eps %>% head(n_st_pr_i[[1]]) %>% t() %>% round(3)
  #(cpt[,,15,drop=FALSE]) %>% round(3)
  #theta_BDeu_eps[(15-1)*n_st_pr_i[[1]] + (1:n_st_pr_i[[1]]),] %>% t() %>% round(3)
  
  data_list_hierMD <- list(
    n_st_ch = n_st_ch, # number of child states
    n_st_pr = prod(dim(cpt)[-1]), # number of total combos of parent states
    N_ch_pr = N_ch_pr, # number of cases at all combos of parents & child
    alpha_0 = array(rep(1,n_st_ch),dim=n_st_ch) # hyperparameter for Dirichlet priors
  )
  
  n_st_pr_i <- {dim(cpt)[-1]} %>% array(dim=length(.))
  data_list_hierMDmix <- c(
    data_list_hierMD,
    list(
      n_parent  = length(pr_names), # number of parents
      n_st_pr_i = n_st_pr_i, # number of states for each parent
      # state of each parent for each combo of parents:
      i_st_pr   = as.matrix(expand.grid(purrr::map(n_st_pr_i,~seq(1,.x)))) 
    )
  )
  
  # Estimate posterior of CPT for child node:
  # 1. hierMD
  sf_hierMD   <- vb( sm_hierMD, data = data_list_hierMD )
  rslt_hierMD <- rstan::extract(sf_hierMD)
  names(rslt_hierMD)
  theta_hierMD <- rslt_hierMD$theta %>% apply(2:length(dim(.)),mean)
  # 2. hierMDmix
  sf_hierMDmix   <- vb( sm_hierMDmix, data = data_list_hierMDmix )
  rslt_hierMDmix <- rstan::extract(sf_hierMDmix)
  names(rslt_hierMDmix)
  theta_hierMDmix <- rslt_hierMDmix$theta %>% apply(2:length(dim(.)),mean)
  
  result <- list(
    theta_BDeu1     = theta_BDeu1,
    theta_BDeu_eps  = theta_BDeu_eps,
    theta_hierMD    = theta_hierMD,
    theta_hierMDmix = theta_hierMDmix
  )
  if(full){
    result <- c(
      result,
      list(
        rslt_hierMD    = rslt_hierMD,
        rslt_hierMDmix = rslt_hierMDmix,
        data_list_hierMD = data_list_hierMD,
        data_list_hierMDmix = data_list_hierMDmix
      )
    )
  }
  invisible( result)
}

rslt_gender <- gen_CPT_hierMD( bbn, df, ch_name = "gender" )
rslt_age    <- gen_CPT_hierMD( bbn, df, ch_name = "age" )
rslt_occ    <- gen_CPT_hierMD( bbn, df, ch_name = "occupation" )

rslt <- list(gender=rslt_gender, age = rslt_age, occupation = rslt_occ)

# Test the gender-node revision
theta_BDeu_eps  <- rslt_gender$theta_BDeu_eps
theta_hierMD    <- rslt_gender$theta_hierMD
theta_hierMDmix <- rslt_gender$theta_hierMDmix

(cpt[,,15,drop=FALSE]) %>% round(3)
theta_BDeu_eps[(15-1)*n_st_pr_i[[1]] + (1:n_st_pr_i[[1]]),] %>% t() %>% round(3) %>%
  {dimnames(.)<- dimnames(cpt)[-3]; .}
theta_hierMD[(15-1)*n_st_pr_i[[1]] + (1:n_st_pr_i[[1]]),] %>% t() %>% round(3) %>%
  {dimnames(.)<- dimnames(cpt)[-3]; .}
theta_hierMDmix[(15-1)*n_st_pr_i[[1]] + (1:n_st_pr_i[[1]]),] %>% t() %>% round(3) %>%
  {dimnames(.)<- dimnames(cpt)[-3]; .}

# REVISION OF BBN ====

# Check effect upon inference
# Revise BBN using CPT from hierMD
ch_name <- "gender"
bbn_hierMD <- bbn
bbn_hierMD$cptlist[[ch_name]] <- array(t(theta_hierMD),dim=dim(cpt),dimnames=dimnames(cpt))
bbn_hierMD <- compile(bbn_hierMD)
# Revise BBN using CPT from hierMDmix
bbn_hierMDmix <- bbn
bbn_hierMDmix$cptlist[[ch_name]] <- array(t(theta_hierMDmix),dim=dim(cpt),dimnames=dimnames(cpt))
bbn_hierMDmix <- compile(bbn_hierMDmix)

revise_bbn <- function(bbn, ch_name, theta_hierMD, theta_hierMDmix){
  bbn_hierMD <- bbn$hierMD
  bbn_hierMD$cptlist[[ch_name]] <- array(t(theta_hierMD),dim=dim(cpt),dimnames=dimnames(cpt))
  bbn_hierMD <- compile(bbn_hierMD)
  # Revise BBN using CPT from hierMDmix
  bbn_hierMDmix <- bbn$hierMDmix
  bbn_hierMDmix$cptlist[[ch_name]] <- array(t(theta_hierMDmix),dim=dim(cpt),dimnames=dimnames(cpt))
  bbn_hierMDmix <- compile(bbn_hierMDmix)
  return( list(hierMD = bbn_hierMD, hierMDmix = bbn_hierMDmix) )
}

bbn_list <- list( hierMD=bbn, hierMDmix=bbn )
for(ch_name in c("age","gender","occupation")){
  
  bbn_list <- revise_bbn(
    bbn_list, 
    ch_name         = ch_name, 
    theta_hierMD    = rslt[[ch_name]]$theta_hierMD, 
    theta_hierMDmix = rslt[[ch_name]]$theta_hierMDmix 
  )
}

bbn_hierMD    <- bbn_list$hierMD
bbn_hierMDmix <- bbn_list$hierMDmix


# IMPACT ON BAYESIAN INFERNCE ====
# Conditional ratings distribution and expected rating given gender,
# posterior given a programmer who has seen the movie.
case_profile <- list(occupation="programmer", Has_Seen = "yes", age = "(33,44]")
querygrain(bbn,
           nodes = c("Str.ctr_79","gender"), 
           evidence = case_profile,
           type = "conditional") %T>% {print(signif(.,3))} %>% 
  {list(Expected_Rating = round(t(.) %*% c(-3:2,0),2))}
querygrain(bbn_hierMD,
           nodes = c("Str.ctr_79","gender"), 
           evidence = case_profile,
           type = "conditional") %T>% {print(signif(.,3))} %>% 
  {list(Expected_Rating = round(t(.) %*% c(-3:2,0),2))}
querygrain(bbn_hierMDmix,
           nodes = c("Str.ctr_79","gender"), 
           evidence = case_profile,
           type = "conditional") %T>% {print(signif(.,3))} %>% 
  {list(Expected_Rating = round(t(.) %*% c(-3:2,0),2))}

# Conditional ratings distribution given gender,
# posterior given a doctor who has seen the movie.
case_profile <- list(occupation="doctor", Has_Seen = "yes", age = "(33,44]")
querygrain(bbn,
           nodes = c("Str.ctr_79","gender"), 
           evidence = case_profile,
           type = "conditional") %T>% {print(signif(.,3))} %>% 
  {list(Expected_Rating = round(t(.) %*% c(-3:2,0),2))}
querygrain(bbn_hierMD,
           nodes = c("Str.ctr_79","gender"), 
           evidence = case_profile,
           type = "conditional") %T>% {print(signif(.,3))} %>% 
  {list(Expected_Rating = round(t(.) %*% c(-3:2,0),2))}
querygrain(bbn_hierMDmix,
           nodes = c("Str.ctr_79","gender"), 
           evidence = case_profile,
           type = "conditional") %T>% {print(signif(.,3))} %>%
  {list(Expected_Rating = round(t(.) %*% c(-3:2,0),2))}


# IMPACT ON GENERALIZED BAYES FACTOR ====
gbf <- function(bbn, gndr = "F",occ = "doctor",age = "(33,44]"){
  # Generalized Bayes Factors: O(H|E)/O(H) -- posterior-to-prior odds-ratio
  # (all conditioned upon Has_Seen = "yes")
  joint <- bbn %>%
    querygrain(
      nodes = c("gender","occupation","age"),
      evidence = list(Has_Seen = "yes" ),
      type  = "joint"
    )
  
  prior_odds_F_occ <- joint %>% {./(1 - .)}
  
  joint_post <- bbn %>%
    querygrain(
      nodes = c("gender","occupation","age"),
      evidence=list(LIKE = "yes"),
      type="joint"
    )
  posterior_odds_F_occ_like <-  joint_post %>% {./(1-.)}
  
  # H: {gender,occupation,age}, E: {like movie} (all given Has_Seen="yes")
  GBF_HE <- posterior_odds_F_occ_like / prior_odds_F_occ
  
  full <- list(
    gbf        = GBF_HE,
    wte        = 10*log10(GBF_HE), # in decibans
    joint      = joint,
    joint_post = joint_post,
    prior_odds = signif(prior_odds_F_occ,3),
    post_odds  = signif(posterior_odds_F_occ_like,3)
  )
  at_cond <- lapply( full, function(x){x[age,occ,gndr]} )
  invisible( list( at_cond = at_cond, full = full ) )
}

# Compare models impact on GBF(H:E) under different case profiles as
# the "hypothesis" H, given the "evidence" E = LIKE = "yes".
list( BDeu = bbn, hierMD = bbn_hierMD, hierMDmix = bbn_hierMDmix ) %>%
  imap_dfr(
    ~ gbf(bbn = .x, gndr = "F", occ = "doctor", age = "(33,44]") %$% 
      map(at_cond,signif,3) %>%
      as_tibble() %>% 
      mutate(model=.y) %>% 
      select(model,everything())
  )

list( BDeu = bbn, hierMD = bbn_hierMD, hierMDmix = bbn_hierMDmix ) %>%
  imap_dfr(
    ~ gbf(bbn = .x, gndr = "M", occ = "programmer", age = "(33,44]") %$% 
      map(at_cond,signif,3) %>%
      as_tibble() %>% 
      mutate(model=.y) %>% 
      select(model,everything())
  )

list( BDeu = bbn, hierMD = bbn_hierMD, hierMDmix = bbn_hierMDmix ) %>%
  imap_dfr(
    ~ gbf(bbn = .x, gndr = "F", occ = "programmer", age = "(33,44]") %$% 
      map(at_cond,signif,3) %>%
      as_tibble() %>% 
      mutate(model=.y) %>% 
      select(model,everything())
  )

list( BDeu = bbn, hierMD = bbn_hierMD, hierMDmix = bbn_hierMDmix ) %>%
  imap_dfr(
    ~ gbf(bbn = .x, gndr = "F", occ = "administrator", age = "(33,44]") %$% 
      map(at_cond,signif,3) %>%
      as_tibble() %>% 
      mutate(model=.y) %>% 
      select(model,everything())
  )

