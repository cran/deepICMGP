# Standalone deepICMGP walkthrough; requires an installed package.
# Uses only deepICMGP and lhs; no comparison models.
library(deepICMGP)
library(lhs)

# Branin: 2 input dimension(s)
f1 <- function(x) {
  x1 <- x[, 1] * 15 - 5
  x2 <- x[, 2] * 15
  10 * sqrt(
    (-1.275 * (1.2 * x1 + 0.4)^2 / pi^2 +
       5 * (1.2 * x1 + 0.4) / pi + (1.2 * x2 + 0.4) - 6)^2 +
      (10 - 5 / (4 * pi)) * cos(1.2 * x1 + 0.4) + 10
  ) + 2 * (1.2 * x1 + 1.9) - 3 * (3 * (1.2 * x2 + 2.4) - 1) -
    1 - 3 * x2 + 1
}
f2 <- function(x) {
  x1 <- x[, 1] * 15 - 5
  x2 <- x[, 2] * 15
  10 * sqrt(
    (-1.275 * (x1 + 2)^2 / pi^2 + 5 * (x1 + 2) / pi + (x2 + 2) - 6)^2 +
      (10 - 5 / (4 * pi)) * cos(x1 + 2) + 10
  ) + 2 * (x1 - 0.5) - 3 * (3 * x2 - 1) - 1
}
f3 <- function(x) {
  x1 <- x[, 1] * 15 - 5
  x2 <- x[, 2] * 15
  (-1.275 * x1^2 / pi^2 + 5 * x1 / pi + x2 - 6)^2 +
    (10 - 5 / (4 * pi)) * cos(x1) + 10
}
response <- function(x) cbind(f1(x), f2(x), f3(x))
set.seed(2025)
X <- maximinLHS(30, 2)
Y <- response(X)

Xtest <- maximinLHS(100, 2)

fit <- fit_deepICMGP(X, Y, nmcmc = 5000, true_g = 1e-8)
trim <- trim_deepICMGP(fit, burn = 2000, thin = 2)
pred <- pred_deepICMGP(trim, Xtest)
alc <- ALC_deepICMGP(trim, Xcand = Xtest)
X_next <- alc$x_next

# Predictions are already in the original response units.
rmse <- sqrt(colMeans((response(Xtest) - pred$mean)^2))
rmse
