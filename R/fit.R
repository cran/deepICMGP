# Function Contents -----------------------------------------------------------
# External: (see documentation below)
#   fit_deepICMGP

#Fit_deepICMGP -----------------------------------------------------------------
#' @title Fit a two-layer multi-output Gaussian process
#' @description Fit a deep intrinsic coregionalization Gaussian process (deepICMGP) to
#' multiple outputs observed at shared input locations. The sampler updates
#' kernel parameters and a latent Gaussian process layer. The coregionalization
#' matrices describe dependence among outputs and among latent coordinates.
#'
#' @param x Numeric input matrix with \eqn{n} rows and \eqn{d} columns. 
#' A numeric vector is converted to a one-column matrix. Each column is 
#' automatically scaled to \eqn{[0,1]} using its training minimum and range.
#' @param y Numeric response matrix with \eqn{n} rows and \eqn{q} columns, 
#' in the same row order as \code{x}. Each column represents one output and 
#' is automatically centered and divided by its training sample standard deviation. 
#' For a single output, supply a one-column matrix, i.e., \code{matrix(y, ncol = 1)}. 
#' @param nmcmc Length of the MCMC chain. Must be an integer greater than 1. 
#' Default is \code{5000} MCMC runs.
#' @param D Positive integer giving the number of latent variables. Defaults to \code{max(ncol(x), ncol(y))}.
#' @param trace Logical. Print progress for every 500 iterations when \code{TRUE}.
#' @param true_g A positive scalar specifying a fixed value for the outer-layer 
#' nugget \code{g}. Default is \code{NULL}, in which case \code{g} is sampled 
#' by MCMC with \code{eps} as its lower bound. A small positive value approximates 
#' a noise-free response model, e.g., \code{true_g = 1e-8}.
#' @param settings List of proposal and prior settings.
#' Default is \code{NULL}, which uses the defaults described in Details.
#' One may provide alternative values for any settings for customizing.
#' @param eps A small positive constant for numerical stability.
#' The default is \code{sqrt(.Machine$double.eps)}, which is approximately \code{1e-8}.
#'
#' @details The model uses squared-exponential kernels with a scalar lengthscale parameter
#' for each layer. The outer layer maps the \eqn{D}-dimensional latent values
#' to the \eqn{q} outputs. The prior settings apply to the model after the inputs are scaled
#' and each output is standardized to mean zero and standard deviation one.
#'
#' Metropolis-Hastings updates are used for the lengthscale parameters and, when
#' not fixed, the nugget. The latent layer is updated by elliptical slice
#' sampling. The proposal for a positive parameter \eqn{a} is uniform on
#' \code{[settings$l * a / settings$u, settings$u * a / settings$l]}.
#' Defaults are \code{l = 1} and \code{u = 2}. Gamma prior densities are
#' evaluated at the parameter minus \code{eps}, using shape and rate:
#' \itemize{
#' \item \code{alpha$g = 1.5}, \code{beta$g = 3.9}.
#' \item \code{alpha$theta_y = 1.5}, \code{beta$theta_y = 3.9 / 6}.
#' \item \code{alpha$theta_w = 1.5}, \code{beta$theta_w = 3.9 / 4}.
#' }
#' where \code{g} is the nugget parameter, \code{theta_y} and \code{theta_w} are the
#' lengthscale parameters for outer and inner-layer, respectively.
#' 
#' For example, \code{settings = list(l = 1, u = 1.5, beta = list(g = 5))}
#' overrides only those entries. Coregionalization matrices are estimated 
#' during MCMC sampling, rather than sampled directly from their posterior distributions.
#'
#' Initial latent values use the first \code{D} scaled input columns when
#' \code{D <= ncol(x)}, or reuse the scaled input matrix otherwise.
#' The inner-layer prior mean is zero, and the inner kernel scale is one. 
#' Input and output transformations are learned from training data only and 
#' retained for prediction and active learning.
#'
#' One can inspect trace plots and convergence before using the samples. We suggest removing
#' burn-in with \code{trim_deepICMGP} before prediction or active learning.
#'
#' @return A list of class \code{deepICMGP}. Let \eqn{nt} be the number of MCMC samples.
#' \itemize{
#' \item \code{x}, \code{y}: the scaled input and standardized response matrices used by the sampler.
#' \item \code{scaling}: training transformations \code{x_min}, \code{x_range}, \code{y_mean}, and \code{y_sd}.
#' \item \code{nmcmc}, \code{settings}, \code{eps}: stored MCMC count and settings.
#' \item \code{g}, \code{theta_y}, \code{theta_w}: length-\eqn{nt} vectors of nugget and lengthscale values.
#' \item \code{B_y}: \eqn{q \times q \times nt} array of outer-layer coregionalization estimates.
#' \item \code{B_w}: \eqn{D \times D \times nt} array of inner-layer coregionalization estimates.
#' \item \code{w}: list of \eqn{nt} latent matrices, each \eqn{n \times D}.
#' \item \code{ll}: length-\eqn{nt} vector of outer-layer log-likelihood values, up to omitted constants.
#' \item \code{time}: elapsed computation time in seconds.
#' }
#'
#' @section Code provenance:
#' This implementation adapts code from the \pkg{deepgp} package and extends
#' it to deep intrinsic coregionalization models. See Sauer, Gramacy, and
#' Higdon (2023) for the underlying deep GP methodology, and
#' \url{https://CRAN.R-project.org/package=deepgp} for the source package.
#'
#' @references Chang, C.-Y. and Sung, C.-L. (2026). Deep Intrinsic Coregionalization Multi-Output Gaussian Process Surrogate with Active Learning. International Journal for Uncertainty Quantification, 16(4), 29--52. \doi{10.1615/Int.J.UncertaintyQuantification.2026062570}.
#'
#'
#' Sauer, A., Gramacy, R. B., and Higdon, D. (2023). Active Learning for Deep Gaussian Process Surrogates. Technometrics, 65(1), 4--18. \doi{10.1080/00401706.2021.2008505}.
#'
#' @seealso \code{\link{trim_deepICMGP}}, \code{\link{pred_deepICMGP}}, \code{\link{ALC_deepICMGP}}
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
#' # Fit a deepICMGP model with 2000 MCMC samples and set nugget to 1e-8 for deterministic case.
#' fit <- fit_deepICMGP(X, Y, nmcmc = 2000, true_g = 1e-8)
#' 
#' # Check the last coregionalization matrices for both layers.
#' fit$B_w[, , 2000]
#' fit$B_y[, , 2000]
#' 
#' # Trace plots for checking convergence of lengthscale parameters.
#' ## Inner-layer lengthscale
#' ts.plot(fit$theta_w)
#' 
#' ## Outer-layer lengthscale
#' ts.plot(fit$theta_y)
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
#' # Fit a deepICMGP model with 5000 MCMC samples and set nugget to 1e-8 for deterministic case.
#' fit <- fit_deepICMGP(X, Y, nmcmc = 5000, true_g = 1e-8)
#' 
#' # Trace plots for checking convergence of lengthscale parameters.
#' ## Inner-layer lengthscale
#' ts.plot(fit$theta_w)
#' 
#' ## Outer-layer lengthscale
#' ts.plot(fit$theta_y)
#' }
#' @export



fit_deepICMGP <- function(x, y, nmcmc = 5000, D = max(ncol(x), ncol(y)),
                        trace = TRUE,
                        true_g = NULL, settings = NULL, eps = sqrt(.Machine$double.eps)) {
  tic <- proc.time()[3]
  x <- numeric_matrix(x, "x", vector_ok = TRUE)
  check_inputs_deepicmgp(x, y, true_g)
  scalar_number(nmcmc, "nmcmc", 2, integer = TRUE)
  scalar_number(D, "D", 1, integer = TRUE)
  scalar_number(eps, "eps", strict = TRUE)
  check_flag(trace, "trace")
  settings <- check_settings_deepicmgp(settings)
  scaling <- list(x_min = apply(x, 2L, min),
                  x_range = apply(x, 2L, max) - apply(x, 2L, min),
                  y_mean = colMeans(y), y_sd = sqrt(apply(y, 2L, var)))
  scaling$x_range[scaling$x_range == 0] <- 1
  x <- scale_inputs(x, scaling)
  y <- sweep(sweep(y, 2L, scaling$y_mean, "-"), 2L, scaling$y_sd, "/")
  numeric_matrix(x, "scaled x")
  numeric_matrix(y, "standardized y")
  out <- list(x=x, y=y, scaling=scaling, nmcmc=as.integer(nmcmc), settings=settings, eps=eps)
  out <- c(out, deepICM(x, y, as.integer(nmcmc), as.integer(D), trace, true_g, settings, eps))
  out$time <- proc.time()[3] - tic
  class(out) <- "deepICMGP"
  out
}
