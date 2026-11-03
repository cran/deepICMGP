# Internal numerical and validation helpers.
#' @importFrom stats cov cor var dgamma rnorm runif
NULL

# Squared Euclidean distances between rows.
sq_dist <- function(X1, X2 = NULL) {
  X1 <- numeric_matrix(as.matrix(X1), "X1")
  X2 <- if (is.null(X2)) X1 else numeric_matrix(as.matrix(X2), "X2")
  if (ncol(X1) != ncol(X2)) stop("dimension of X1 & X2 do not match")
  distance <- matrix(0, nrow(X1), nrow(X2))
  for (j in seq_len(ncol(X1)))
    distance <- distance + outer(X1[,j], X2[,j], "-")^2
  distance
}

kernel_icm <- function(distance, tau2, theta, g) {
  k <- tau2 * exp(-distance / theta)
  if (g != 0) {
    if (nrow(k) != ncol(k)) stop("nugget requires a square kernel")
    diag(k) <- diag(k) + tau2 * g
  }
  k
}

invdet_icm <- function(x) {
  r <- chol(x)
  list(Mi = chol2inv(r), ldet = 2 * sum(log(diag(r))))
}

scalar_number <- function(x, name, lower = 0, integer = FALSE, strict = FALSE) {
  if (!is.numeric(x) || length(x) != 1L || !is.finite(x) ||
      (if (strict) x <= lower else x < lower) ||
      (integer && (x != floor(x) || x > .Machine$integer.max)))
    stop(name, " must be a finite ", if (integer) "integer " else "number ",
         if (strict) "greater than " else "at least ", lower, call. = FALSE)
  invisible(x)
}

check_flag <- function(x, name) {
  if (!is.logical(x) || length(x) != 1L || is.na(x))
    stop(name, " must be TRUE or FALSE", call. = FALSE)
}

numeric_matrix <- function(x, name, vector_ok = FALSE) {
  if (vector_ok && is.numeric(x) && is.null(dim(x))) x <- matrix(x, ncol = 1L)
  if (!is.matrix(x) || !is.numeric(x) || any(dim(x) == 0L) || any(!is.finite(x)))
    stop(name, " must be a nonempty finite numeric matrix", call. = FALSE)
  x
}

check_object <- function(object, min_states = 1L) {
  if (!inherits(object, "deepICMGP")) stop("object must be a deepICMGP fit")
  scalar_number(object$nmcmc, "object$nmcmc", min_states, integer = TRUE)
  n <- nrow(object$x)
  D <- ncol(object$w[[1L]])
  q <- ncol(object$y)
  nt <- object$nmcmc
  if (length(object$w) != nt || any(vapply(object$w, function(w)
      !is.matrix(w) || !identical(dim(w), c(n, D)) || any(!is.finite(w)), logical(1))) ||
      !identical(dim(object$B_y), c(q, q, as.integer(nt))) ||
      !identical(dim(object$B_w), c(D, D, as.integer(nt))) ||
      any(vapply(object[c("g", "theta_y", "theta_w", "ll")], length, integer(1)) != nt))
    stop("inconsistent MCMC dimensions in object")
  invisible(object)
}

clear_predictions <- function(object) {
  object[c("x_new", "mean", "s2", "C2", "s2_all", "w_new", "quad")] <- NULL
  object
}

regularize_B <- function(B) {
  B <- as.matrix(B)
  # Equivalent to the original correlation-scale 0.01 diagonal adjustment.
  diag(B) <- diag(B) + 0.01 * diag(B)
  B
}

# Apply training transformations; prediction inputs stay in their original units
# on the returned object.
scale_inputs <- function(x, scaling) {
  sweep(sweep(x, 2L, scaling$x_min, "-"), 2L, scaling$x_range, "/")
}
