# Function Contents -----------------------------------------------------------
# External: (see documentation below)
#   trim_deepICMGP

# Trim deepICMGP --------------------------------------------------------------
#' @title Remove burn-in and thin stored MCMC samples
#' @description Retain selected samples from a fitted deepICMGP object before prediction or active learning.
#'
#' @param object An object of class \code{deepICMGP} returned by
#' \code{fit_deepICMGP} or \code{trim_deepICMGP}.
#' @param burn Nonnegative integer smaller than \code{object$nmcmc}.
#' @param thin Positive integer thinning interval. Retains original sample indices divisible by \code{thin}.
#' When \code{thin = 1}, it retains every sample after burn-in.
#'
#' @details The number of stored samples \eqn{nt} must be greater than
#' \code{burn}. After discarding burn-in, keep samples at the thinning interval.
#' When both \eqn{nt} and \code{burn} are multiples of \code{thin},
#' the retained count is \code{(nt - burn) / thin}.
#' For example, 1000 samples with \code{burn = 500} and \code{thin = 2}
#' leave 250 samples. The actual retained count is stored in \code{nmcmc}.
#'
#' Retain at least two samples for prediction, which estimates variance across samples.
#' Invalid burn-in/thinning values and choices that retain no samples are rejected.
#'
#' @return An object of class \code{deepICMGP} with \code{g}, \code{theta_y},
#' \code{theta_w}, \code{B_y}, \code{B_w}, \code{w}, and \code{ll}
#' subset to the retained samples, and \code{nmcmc} updated accordingly.
#' Elapsed trimming time is added to \code{time}. Training data and settings are unchanged.
#'
#' @references Chang, C.-Y. and Sung, C.-L. (2026). Deep Intrinsic Coregionalization Multi-Output Gaussian Process Surrogate with Active Learning. International Journal for Uncertainty Quantification, 16(4), 29--52. \doi{10.1615/Int.J.UncertaintyQuantification.2026062570}.
#'
#' Sauer, A., Gramacy, R. B., and Higdon, D. (2023). Active Learning for Deep Gaussian Process Surrogates. Technometrics, 65(1), 4--18. \doi{10.1080/00401706.2021.2008505}.
#'
#' @seealso \code{\link{fit_deepICMGP}}, \code{\link{pred_deepICMGP}}
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
#' # Fit a deepICMGP model with 2000 MCMC samples and set nugget to 1e-8 
#' # for deterministic case.
#' fit <- fit_deepICMGP(X, Y, nmcmc = 2000, true_g = 1e-8)
#' 
#' # Trim a fitted deepICMGP model by removing first 1500 MCMC samples and
#' # retaining every second sample.
#' trim <- trim_deepICMGP(fit, burn = 1500, thin = 2)
#' 
#' # Check the number of retained MCMC samples.
#' trim$nmcmc
#' @export


trim_deepICMGP <- function(object, burn, thin = 1) {
  tic <- proc.time()[3]
  check_object(object)
  scalar_number(burn, "burn", integer=TRUE)
  scalar_number(thin, "thin", 1, integer=TRUE)
  if (burn >= object$nmcmc) stop("burn must be less than nmcmc")
  indx <- which(seq_len(object$nmcmc) > burn & seq_len(object$nmcmc) %% thin == 0)
  if (!length(indx)) stop("burn and thin retain no MCMC samples")
  object <- clear_predictions(object)
  object$nmcmc <- length(indx)
  for (name in c("g", "theta_y", "theta_w", "ll")) object[[name]] <- object[[name]][indx]
  object$B_y <- object$B_y[,,indx,drop=FALSE]
  object$B_w <- object$B_w[,,indx,drop=FALSE]
  object$w <- object$w[indx]
  object$time <- object$time + proc.time()[3] - tic
  object
}
