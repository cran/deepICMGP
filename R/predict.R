# Function Contents -----------------------------------------------------------
# External: (see documentation below)
#   pred_deepICMGP


#ICM_two_layer ---------------------------------------------------------------
#' @title Predict output means, variances, and within-location covariances
#' @description Compute posterior predictive mean, variance and within-location
#' output covariances at new inputs using all retained MCMC samples. Parallel computation
#' for prediction is supported with the \pkg{parallel} package.
#'
#' @param object An object of class \code{deepICMGP} returned by
#' \code{fit_deepICMGP} or \code{trim_deepICMGP}, containing at least two
#' retained MCMC samples.
#' @param x_new Numeric matrix with \code{n_new = nrow(x_new)} rows and the \eqn{d} columns, 
#' using the same column order and original units as the training inputs. For 
#' one input dimension a numeric vector is accepted. For one location in multiple
#' dimensions, please use a one-row matrix.
#' @param store_latent Logical. Include latent predictions and kernel quadratic terms 
#' in the returned prediction object. Default is \code{FALSE}.
#' @param return_all Logical. Return the pointwise variance matrix for each retained 
#' sample in \code{s2_all}. Default is \code{FALSE}.
#' @param cores Number of CPU cores for parallel prediction using the \pkg{parallel} package. 
#' Must be a positive integer. Default is \code{1}.
#'
#' @details The function samples latent values at the new inputs, then combines
#' conditional moments of outer-layer using the laws of total expectation and covariance.
#'
#' \code{C2[i, , ]} describes dependence among outputs at location \code{i}.
#' \code{s2_all} stores conditional variances for individual MCMC samples.
#' \code{s2} combines their average with the variation in predicted
#' means across samples to obtain the posterior predictive variance.
#'
#'
#' @return The supplied \code{deepICMGP} object with the following fields provided.
#' Let \eqn{nt} be the number of retained samples and \eqn{q} the number of outputs.
#' The number of prediction locations is \code{n_new = nrow(x_new)}.
#' \itemize{
#' \item \code{x_new}: the \eqn{n_{\mathrm{new}} \times d}{n_new x d} prediction matrix.
#' \item \code{mean}: \eqn{n_{\mathrm{new}} \times q}{n_new x q} matrix of posterior predictive means.
#' \item \code{s2}: \eqn{n_{\mathrm{new}} \times q}{n_new x q} matrix of posterior predictive variances.
#' \item \code{C2}: \eqn{n_{\mathrm{new}} \times q \times q}{n_new x q x q} array of within-location output covariances. \code{C2[i, , ]} is the covariance matrix at row \code{i}.
#' \item \code{s2_all}: when \code{return_all = TRUE}, a list of \eqn{nt} variance matrices, each \eqn{n_{\mathrm{new}} \times q}{n_new x q}.
#' \item \code{w_new}: when \code{store_latent = TRUE}, a list of \eqn{nt} sampled latent matrices, each \eqn{n_{\mathrm{new}} \times D}{n_new x D}.
#' \item \code{quad}: when \code{store_latent = TRUE}, an \eqn{n_{\mathrm{new}} \times nt}{n_new x nt} matrix of outer-kernel quadratic terms used for active learning.
#' \item \code{time}: Total elapsed computation time in seconds, including the time 
#' stored in \code{object} and the current prediction call.
#' }
#' 
#'
#' @section Code provenance:
#' This implementation adapts code from the \pkg{deepgp} package and extends
#' it to deep intrinsic coregionalization models. See Sauer, Gramacy, and
#' Higdon (2023) for the underlying deep GP methodology, and
#' \url{https://CRAN.R-project.org/package=deepgp} for the source package.
#'
#' @references Chang, C.-Y. and Sung, C.-L. (2026). Deep Intrinsic Coregionalization Multi-Output Gaussian Process Surrogate with Active Learning. International Journal for Uncertainty Quantification, 16(4), 29--52. \doi{10.1615/Int.J.UncertaintyQuantification.2026062570}.
#'
#' Sauer, A., Gramacy, R. B., and Higdon, D. (2023). Active Learning for Deep Gaussian Process Surrogates. Technometrics, 65(1), 4--18. \doi{10.1080/00401706.2021.2008505}.
#'
#' @seealso \code{\link{fit_deepICMGP}}, \code{\link{trim_deepICMGP}}, \code{\link{ALC_deepICMGP}}
#'
#' @examples
#' library(lhs)
#'
#' # Forrester Function: 1d input with 2 outputs
#' f1 <- function(x) {
#'   (6 * x - 2)^2 * sin(12 * x - 4)
#' }
#' f2 <- function(x) {
#'   0.5 * f1(x) + 10 * (x - 0.5) + 5
#' }
#' response <- function(x) cbind(f1(x), f2(x))
#' set.seed(2025)
#' X <- maximinLHS(9, 1)
#' Y <- response(X)
#'
#' Xtest <- maximinLHS(50, 1)
#' Ytrue <- response(Xtest)
#'
#' # Fit a deepICMGP model with 2000 MCMC samples and set nugget to 1e-8
#' # for deterministic case.
#' fit <- fit_deepICMGP(X, Y, nmcmc = 2000, true_g = 1e-8)
#' 
#' # Trim a fitted deepICMGP model by removing first 1500 MCMC samples and
#' # retaining every second sample.
#' trim <- trim_deepICMGP(fit, burn = 1500, thin = 2)
#' 
#' # Predict a trimmed deepICMGP model on Xtest.
#' pred <- pred_deepICMGP(trim, x_new = Xtest)
#' 
#' # Calculate RMSE for each output.
#' RMSE <- numeric(ncol(Y))
#' for (j in seq_len(ncol(Y))) {
#'   RMSE[j] <- sqrt(mean((pred$mean[, j] - Ytrue[, j])^2))
#' }
#' RMSE
#'
#' \donttest{
#' # Branin Function: 2d input with 3 outputs
#' f1 <- function(x) {
#'   x1 <- x[, 1] * 15 - 5
#'   x2 <- x[, 2] * 15
#'   10 * sqrt(
#'     (-1.275 * (1.2 * x1 + 0.4)^2 / pi^2 +
#'        5 * (1.2 * x1 + 0.4) / pi + (1.2 * x2 + 0.4) - 6)^2 +
#'       (10 - 5 / (4 * pi)) * cos(1.2 * x1 + 0.4) + 10
#'   ) + 2 * (1.2 * x1 + 1.9) - 3 * (3 * (1.2 * x2 + 2.4) - 1) -
#'     1 - 3 * x2 + 1
#' }
#' f2 <- function(x) {
#'   x1 <- x[, 1] * 15 - 5
#'   x2 <- x[, 2] * 15
#'   10 * sqrt(
#'     (-1.275 * (x1 + 2)^2 / pi^2 + 5 * (x1 + 2) / pi + (x2 + 2) - 6)^2 +
#'       (10 - 5 / (4 * pi)) * cos(x1 + 2) + 10
#'   ) + 2 * (x1 - 0.5) - 3 * (3 * x2 - 1) - 1
#' }
#' f3 <- function(x) {
#'   x1 <- x[, 1] * 15 - 5
#'   x2 <- x[, 2] * 15
#'   (-1.275 * x1^2 / pi^2 + 5 * x1 / pi + x2 - 6)^2 +
#'     (10 - 5 / (4 * pi)) * cos(x1) + 10
#' }
#' response <- function(x) cbind(f1(x), f2(x), f3(x))
#' set.seed(2025)
#' X <- maximinLHS(30, 2)
#' Y <- response(X)
#'
#' Xtest <- maximinLHS(100, 2)
#'
#' # Fit a deepICMGP model with 5000 MCMC samples and set nugget to 1e-8
#' # for deterministic case.
#' fit <- fit_deepICMGP(X, Y, nmcmc = 5000, true_g = 1e-8)
#' 
#' # Predict a fitted deepICMGP model on Xtest and return sampled latent values.
#' pred <- pred_deepICMGP(fit, Xtest, store_latent = TRUE)
#' length(pred$w_new)
#' 
#' # For parallel prediction:
#' pred <- pred_deepICMGP(fit, Xtest, cores = 2)
#' }
#' @export


pred_deepICMGP <- function(object, x_new, store_latent = FALSE,
                            return_all = FALSE, cores = 1) {
  tic <- proc.time()[3]
  check_object(object, 2L)
  x_new <- numeric_matrix(x_new, "x_new", vector_ok = TRUE)
  if (ncol(x_new) != ncol(object$x)) stop("x_new must have the same number of columns as x")
  check_flag(store_latent, "store_latent")
  check_flag(return_all, "return_all")
  scalar_number(cores, "cores", 1, integer = TRUE)
  object <- clear_predictions(object)
  object$x_new <- x_new
  x_new <- scale_inputs(x_new, object$scaling)
  numeric_matrix(x_new, "scaled x_new")
  n_new <- nrow(x_new)
  q <- ncol(object$y)
  D <- ncol(object$w[[1L]])
  nt <- object$nmcmc
  dx <- sq_dist(object$x)
  cross <- sq_dist(x_new, object$x)
  # Generate the same random latent innovations regardless of worker count.
  z <- lapply(seq_len(nt), function(t) matrix(rnorm(n_new * D), n_new, D))
  task <- function(t) {
    w <- object$w[[t]]
    k <- krig_icm(w, dx, NULL, cross, object$theta_w[t], object$eps,
                  object$B_w[,,t], W=TRUE)
    w_new <- (z[[t]] * sqrt(k$sig2)) %*% chol(matrix(object$B_w[,,t], D, D)) + k$mean
    k <- krig_icm(object$y, sq_dist(w), NULL, sq_dist(w_new, w),
                  object$theta_y[t], object$g[t], object$B_y[,,t])
    list(mu=k$mean, variance=k$s2, covariance=k$C_new,
         w=if (store_latent) w_new else NULL, quad=if (store_latent) k$quad else NULL)
  }
  if (cores == 1L) results <- lapply(seq_len(nt), task) else {
    cl <- parallel::makePSOCKcluster(min(as.integer(cores), nt))
    on.exit(parallel::stopCluster(cl), add=TRUE)
    # Workers use the same package library as the calling R session.
    parallel::clusterCall(cl, function(paths) .libPaths(paths), .libPaths())
    results <- parallel::parLapply(cl, seq_len(nt), task)
  }
  means <- array(unlist(lapply(results, `[[`, "mu")), c(n_new, q, nt))
  object$mean <- matrix(apply(means, c(1L,2L), mean), n_new, q)
  conditional <- Reduce(`+`, lapply(results, `[[`, "covariance")) / nt
  object$C2 <- conditional
  for (i in seq_len(n_new)) {
    mu <- matrix(means[i,,,drop=FALSE], q, nt)
    centered <- sweep(mu, 1L, object$mean[i,], "-")
    object$C2[i,,] <- matrix(conditional[i,,,drop=FALSE], q, q) + tcrossprod(centered)/(nt-1)
  }
  object$mean <- sweep(sweep(object$mean, 2L, object$scaling$y_sd, "*"),
                       2L, object$scaling$y_mean, "+")
  covariance_scale <- outer(object$scaling$y_sd, object$scaling$y_sd)
  for (i in seq_len(n_new))
    object$C2[i,,] <- matrix(object$C2[i,,,drop=FALSE], q, q) * covariance_scale
  object$s2 <- matrix(0, n_new, q)
  for (i in seq_len(n_new)) object$s2[i,] <- diag(matrix(object$C2[i,,,drop=FALSE],q,q))
  if (store_latent) {
    object$w_new <- lapply(results, `[[`, "w")
    object$quad <- matrix(unlist(lapply(results, `[[`, "quad")), n_new, nt)
  }
  if (return_all) object$s2_all <- lapply(results, function(result)
    sweep(result$variance, 2L, object$scaling$y_sd^2, "*"))
  object$time <- object$time + proc.time()[3] - tic
  object
}
