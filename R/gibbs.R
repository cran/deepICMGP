# Function Contents -----------------------------------------------------------
# Internal:
#   deepICM: Gibbs sampling for deepICMGP

#ICM_two_layer ----------------------------------------------------------------
deepICM <- function(x, y, nmcmc, D, trace, true_g, settings,
                    eps=sqrt(.Machine$double.eps)) {
  d <- ncol(x)
  q <- ncol(y)
  n <- nrow(y)

  g <- vector(length = nmcmc)
  g[1] <- if (is.null(true_g)) 0.01 else true_g
  theta_y <- vector(length = nmcmc)
  theta_y[1] <- 0.1

  B_y <- array(0, c(q,q,nmcmc))


  B_y[, , 1] <- cov(y)
  theta_w <- vector(length = nmcmc)
  theta_w[1] <- 0.5
  w <- vector("list", nmcmc)
  B_w <- array(0, c(D, D, nmcmc))
  if(D <= d){
    w[[1]] <- matrix(x[,1:D], ncol=D)
    if (D == 1){
      B_w[, , 1]<- var(w[[1]])
    } else B_w[, , 1] <- cor(w[[1]])
  }else{
    w[[1]] <- x[, rep(seq_len(d), length.out = D), drop = FALSE]
    B_w[, , 1] <- matrix(0.9,D,D) + diag(0.11,D)
  }

  ll_store <- vector(length = nmcmc)
  ll_store[1] <- NA
  ll_outer <- NA

  dx <- sq_dist(x)
  dw <- sq_dist(w[[1]])


  for (j in 2:nmcmc) {
    if (trace && j %% 500 == 0) {
      message("iteration = ", j)
    }

    # Sample nugget (g)
    if (is.null(true_g)) {

      samp <- sample_g_icm(y, dw, g[j - 1], theta_y[j - 1],
                           alpha = settings$alpha$g, beta = settings$beta$g,
                           l = settings$l, u = settings$u, ll_prev = ll_outer,
                           W=FALSE, eps=eps)
      g[j] <- samp$g
      ll_outer <- samp$ll
    } else g[j] <- true_g



    # Sample outer length scale (theta_y)
    samp <- sample_theta_icm(y, dw, g[j], theta_y[j - 1], B_y[, , j-1],
                             alpha = settings$alpha$theta_y,
                             beta = settings$beta$theta_y, l = settings$l,
                             u = settings$u, ll_prev = ll_outer, W=FALSE, eps=eps)
    theta_y[j] <- samp$theta
    ll_outer <- samp$ll
    B_y[, , j] <- samp$B

    # Sample inner length scale (theta_w)
    samp <- sample_theta_icm(w[[j - 1]], dx, g = eps, theta_w[j - 1], B_w[, , j-1],
                             alpha = settings$alpha$theta_w,
                             beta = settings$beta$theta_w, l = settings$l,
                             u = settings$u, W=TRUE,
                             prior_mean = 0,
                             eps=eps)
    theta_w[j] <- samp$theta
    B_w[, , j] <- samp$B

    # Sample hidden Gaussian layer (w)
    samp <- sample_w_icm(y, w[[j - 1]], dw, dx, g[j], theta_y[j], theta_w[j], B_w[, , j],
                         ll_prev = ll_outer, W=FALSE, prior_mean = 0,
                         inner_tau = rep(1, D), eps=eps)
    w[[j]] <- samp$w
    ll_outer <- samp$ll
    ll_store[j] <- ll_outer
    dw <- samp$dw
  } # end of j for loop

  return(list(g = g, theta_y = theta_y, theta_w = theta_w, B_y = B_y, B_w = B_w,
              w = w, ll = ll_store))
}
