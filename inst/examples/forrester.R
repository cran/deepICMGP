# Standalone deepICMGP walkthrough; requires an installed package.
# Uses only deepICMGP and lhs; no comparison models.
library(deepICMGP)
library(lhs)

# Forrester et al. (2008): 1 input dimension(s)
f1 <- function(x) {
  (6 * x - 2)^2 * sin(12 * x - 4)
}
f2 <- function(x) {
  0.5 * f1(x) + 10 * (x - 0.5) + 5
}
response <- function(x) cbind(f1(x), f2(x))
set.seed(2025)
X <- maximinLHS(9, 1)
Y <- response(X)

Xtest <- maximinLHS(100, 1)

fit <- fit_deepICMGP(X, Y, nmcmc = 5000, true_g = 1e-8)
trim <- trim_deepICMGP(fit, burn = 2000, thin = 2)
pred <- pred_deepICMGP(trim, Xtest)
alc <- ALC_deepICMGP(trim, Xcand = Xtest)
X_next <- alc$x_next

# Predictions are already in the original response units.
rmse <- sqrt(colMeans((response(Xtest) - pred$mean)^2))
rmse
