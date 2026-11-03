check_settings_deepicmgp <- function(settings, D = NULL, n = NULL) {
  if (is.null(settings)) settings <- list()
  if (!is.list(settings)) stop("settings must be a list")
  if (is.null(settings$l)) settings$l <- 1
  if (is.null(settings$u)) settings$u <- 2
  scalar_number(settings$l, "settings$l", strict = TRUE)
  scalar_number(settings$u, "settings$u", strict = TRUE)
  if (settings$l >= settings$u) stop("settings$l must be less than settings$u")
  for (group in c("alpha", "beta")) {
    if (is.null(settings[[group]])) settings[[group]] <- list()
    if (!is.list(settings[[group]])) stop("settings$", group, " must be a list")
  }
  rates <- c(g = 3.9, theta_w = 3.9/4, theta_y = 3.9/6)
  for (name in names(rates)) {
    if (is.null(settings$alpha[[name]])) settings$alpha[[name]] <- 1.5
    if (is.null(settings$beta[[name]])) settings$beta[[name]] <- rates[[name]]
    scalar_number(settings$alpha[[name]], paste0("settings$alpha$",name), strict=TRUE)
    scalar_number(settings$beta[[name]], paste0("settings$beta$",name), strict=TRUE)
  }
  settings
}

check_inputs_deepicmgp <- function(x, y, true_g) {
  numeric_matrix(x, "x")
  numeric_matrix(y, "y")
  if (nrow(x) < 2L || nrow(x) != nrow(y)) stop("x and y must have the same number of rows, at least two")
  if (any(apply(y, 2L, var) <= 0)) stop("each response column must vary")
  if (!is.null(true_g)) scalar_number(true_g, "true_g", strict = TRUE)
  invisible(NULL)
}
