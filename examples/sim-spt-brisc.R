library(fields)
library(BRISC)

##--- simulate a separable spatiotemporal Gaussian field (as in sim-spt.R) ----

nx <- 100
ny <- 100
cov_model <- "Exponential"
range     <- .5   ## fields' aRange
sig       <- 1    ## marginal sd of the spatial process (sigma.sq.true = sig^2)

grid <- list(x = seq(0, 5, len = nx),
             y = seq(0, 5, len = ny))

obj <- circulantEmbeddingSetup(grid,
                               Covariance = cov_model,
                               aRange = range)

set.seed(223)

nt <- 50
## AR(1) parameter
temp_corr <- .6

rf_spt <- vector(mode = "list", length = nt)
## draw t = 1 from the AR(1)'s stationary distribution (Var = sig^2/(1 - temp_corr^2)),
## exactly as BRISC_ST_estimation's exact-AR(1) temporal model assumes; sim-spt.R itself
## starts from Var = sig^2, which is only the stationary variance in the limit of large t
rf_spt[[1]] <- (sig / sqrt(1 - temp_corr^2)) * circulantEmbedding(obj)

for (i in seq_len(nt)[-1]) {
  rf_spt[[i]] <- temp_corr * rf_spt[[i - 1]] +
    sig * circulantEmbedding(obj)
}

## a couple of realizations, for a sanity check
set.panel(2, 2)
for (t in c(1, 2, 19, 20)) {
  image.plot(grid[[1]], grid[[2]], rf_spt[[t]], xlab = "x", ylab = "y")
  title(sprintf("t = %d", t))
}

##--- reshape into BRISC_ST_estimation's N x T layout ----
## coords: N x 2, rows ordered the same way `as.vector()` flattens a
## `grid$x`-by-`grid$y` matrix (x varies fastest); Y: N x T, one column per
## time point, rows aligned with coords.

coords <- as.matrix(expand.grid(x = grid[[1]], y = grid[[2]]))
N <- nrow(coords)

field <- sapply(rf_spt, as.vector)             ## N x nt, noise-free process

## add a small nugget (measurement error) and an intercept, since a purely
## noise-free realization is a degenerate edge case for tau.sq
beta0.true   <- 5
tau.sq.true  <- 0.05
Y <- beta0.true + field + matrix(rnorm(N * nt, sd = sqrt(tau.sq.true)), N, nt)

X <- array(1, dim = c(N, nt, 1))               ## intercept-only design

##--- fit the separable spatiotemporal MLE ----
## fields' exponential covariance uses `range`; BRISC's `phi` is the inverse
## range, so phi.true = 1/range.

t0 <- proc.time()
fit <- BRISC_ST_estimation(coords,
                           c(Y), ## stacking Y
                           X = X,
                           sigma.sq = 1, tau.sq = 0.1, phi = 1,
                           gamma = 0.3, n.neighbors = 15, cov.model = "exponential",
                           verbose = TRUE)
t1 <- proc.time()

cat("\nEstimation time:\n"); print(t1 - t0)

cat("\nEstimated Theta:\n"); print(fit$Theta)
cat("\nTrue parameters:\n")
print(c(sigma.sq = sig^2, tau.sq = tau.sq.true, phi = 1 / range, gamma = temp_corr))

cat("\nEstimated Beta (intercept):\n"); print(fit$Beta)
cat("True beta0:", beta0.true, "\n")

cat("\nLog-likelihood at the MLE:", fit$log_likelihood, "\n")

## Note on gamma: strong spatial correlation (range = 0.5 on a 5x5 domain)
## means the N = 2500 sites behave like far fewer independent replicates of
## the T = 20-step AR(1) series, so a single realization's *empirical*
## lag-1 correlation can reasonably differ from the nominal temp_corr -- the
## MLE is estimating gamma correctly for this particular dataset, not for
## the DGP's parameter in expectation over infinitely many realizations.
lag1 <- sapply(seq_len(N), function(i) cor(field[i, -nt], field[i, -1]))
cat("\n(for reference) mean per-site empirical lag-1 correlation of the",
    "noise-free field:", mean(lag1), "\n")
