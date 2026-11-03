# Function Contents -----------------------------------------------------------
# Internal:
#   krig_icm: calculates posterior mean, s2, and covariance for each testing point

# Krig ------------------------------------------------------------------------
krig_icm <- function(y, dx, d_new = NULL, d_cross = NULL, theta, g, B, mean = TRUE,
                     W = FALSE, prior_mean = 0, prior_mean_new = 0, scale = 1) {
  q <- ncol(y)
  n_new <- nrow(d_cross)
  B <- matrix(B, q, q)
  K <- scale * kernel_icm(dx, 1, theta, g)
  cross <- scale * kernel_icm(d_cross, 1, theta, 0)
  inv <- invdet_icm(K)$Mi
  out <- list()
  if (mean) out$mean <- prior_mean_new + cross %*% inv %*% (y - prior_mean)
  out$quad <- rowSums((cross %*% inv) * cross)
  out$sig2 <- pmax(0, scale * (1 + g) - out$quad)
  out$B <- B
  if (!W) {
    out$C_new <- array(outer(out$sig2, as.vector(B)), c(n_new, q, q))
    out$s2 <- outer(out$sig2, diag(B))
  }
  out
}
