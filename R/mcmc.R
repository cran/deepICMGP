# Function Contents -----------------------------------------------------------
# Internal:
#   logl_icm: evaluates log likelihood with zero mean
#   sample_g_icm: Metropolis Hastings sampling for nugget
#   sample_theta_icm: Metropolis Hastings sampling for theta
#   sample_w_icm: Elliptical Slice Sampling for w layer

# Log Likelihood --------------------------------------------------------------
logl_icm <- function(out_mat, in_dmat, g, theta, W, mu = 0, scale = 1, eps=sqrt(.Machine$double.eps)){

  n <- nrow(out_mat)
  q <- ncol(out_mat)

  K <- scale * kernel_icm(in_dmat, 1, theta, g)
  K_inv <- invdet_icm(K)
  K_det <- K_inv$ldet

  residual <- out_mat - mu
  B <- regularize_B(crossprod(residual, K_inv$Mi %*% residual) / n)

  B_det <- invdet_icm(B)$ldet

  logl <-  - (0.5 * q) * K_det - 0.5 * n * B_det


  return(list(logl = c(logl), B = B))

}

# Sample g --------------------------------------------------------------------

sample_g_icm <- function(out_mat, in_dmat, g_t, theta, alpha, beta, l, u, ll_prev = NA, W=FALSE, eps) {

  # Propose value
  q <- ncol(out_mat)
  g_star <- runif(1, min = l * g_t / u, max = u * g_t / l)

  # Compute acceptance threshold
  ru <- runif(1, min = 0, max = 1)
  if (is.na(ll_prev)){
    ll_prev <- logl_icm(out_mat, in_dmat, g_t, theta, W)$logl
  }

  lpost_threshold <-  ll_prev + log(ru)  + dgamma(g_t - eps, alpha, beta, log = TRUE) - log(g_t) + log(g_star)

  ll_new <- logl_icm(out_mat, in_dmat, g_star, theta, W)$logl


  # Accept or reject (lower bound of eps)
  new <- ll_new + dgamma(g_star - eps, alpha, beta, log = TRUE)

  if (new > lpost_threshold) {
    return(list(g = g_star, ll = ll_new))
  } else {
    return(list(g = g_t, ll = ll_prev))
  }
}

# Sample Theta ----------------------------------------------------------------

sample_theta_icm <- function(out_mat, in_dmat, g, theta_t, B_prev, alpha, beta, l, u,
                             ll_prev = NA, W=FALSE, prior_mean = 0, scale = 1, eps) {

  theta_star <- runif(1, min = l * theta_t / u, max = u * theta_t / l)

  # Compute acceptance threshold
  ru <- runif(1, min = 0, max = 1)
  if (is.na(ll_prev)) {
    ll_prev <- logl_icm(out_mat, in_dmat, g, theta_t, mu = prior_mean, scale = scale, W)$logl
  }

  lpost_threshold <- ll_prev + log(ru) +
    dgamma(theta_t - eps, alpha, beta, log = TRUE) - log(theta_t) + log(theta_star)


  ll_new <- logl_icm(out_mat, in_dmat, g, theta_star, mu = prior_mean, scale = scale, W)

  # Accept or reject (lower bound of eps)
  new <- ll_new$logl + dgamma(theta_star - eps, alpha, beta, log = TRUE)

  if (new > lpost_threshold) {
    return(list(theta = theta_star, ll = ll_new$logl, B = ll_new$B))
  } else {
    return(list(theta = theta_t, ll = ll_prev, B = B_prev))
  }
}

# Elliptical Slice W ----------------------------------------------------------

sample_w_icm <- function(out_mat, w_t, w_t_dmat, in_dmat, g, theta_y,  theta_w,
                         B_w, ll_prev = NA, W=FALSE,
                         prior_mean = matrix(0, nrow = nrow(w_t), ncol = ncol(w_t)),
                         scale = 1, eps, inner_tau) {

  q <- length(theta_y)
  D <- ncol(w_t)

  if (is.na(ll_prev)) {
    ll_prev <- logl_icm(out_mat, w_t_dmat, g, theta_y, W)$logl
  }

  # Draw from prior distribution
  K <- scale * kernel_icm(in_dmat, 1, theta_w, g=eps)


  w_prior <- kronecker(t(chol(matrix(B_w, D, D))), t(chol(K))) %*% rnorm(D*nrow(K))

  # Initialize a and bounds on a
  a <- runif(1, min = 0, max = 2 * pi)
  amin <- a - 2 * pi
  amax <- a

  # Compute acceptance threshold - based on all dimensions of previous w
  ru <- runif(1, min = 0, max = 1)
  ll_threshold <- ll_prev + log(ru)

  # Calculate proposed values, accept or reject, repeat if necessary
  accept <- FALSE
  count <- 0
  w_prev <- as.vector(w_t - prior_mean) # store for re-proposal

  while (accept == FALSE) {
    # while (accept == FALSE & count < 100) {
    count <- count + 1

    # Calculate proposed values and new likelihood
    w_t <- prior_mean + matrix(w_prev * cos(a) + w_prior * sin(a), ncol=D)
    dw <- sq_dist(w_t)

    new_logl <- logl_icm(out_mat, dw, g, theta_y, W)$logl

    # Accept or reject
    if (new_logl > ll_threshold) {
      ll_prev <- new_logl
      accept <- TRUE
    } else {
      # update the bounds on a and repeat
      if (a < 0) {
        amin <- a
      } else {
        amax <- a
      }
      a <- runif(1, amin, amax)
      if (count > 100) {
        stop('reached maximum iterations of ESS')
        # warning('reached maximum iterations of ESS')
      }
    } # end of else statement
  } # end of while loop

  return(list(w = w_t, ll = ll_prev, dw = dw))
}
