library(fields)
library(BRISC)

##--- simulate a small separable spatiotemporal Gaussian field ----

nx <- 20
ny <- 20
cov_model  <- "Matern"
range      <- .5
smoothness <- 1
sig        <- 1

grid <- list(x = seq(0, 5, len = nx),
             y = seq(0, 5, len = ny))

obj <- circulantEmbeddingSetup(grid,
                               Covariance = cov_model,
                               aRange = range,
                               smoothness = smoothness)

set.seed(223)

nt <- 8
temp_corr <- .6

rf_spt <- vector(mode = "list", length = nt)
rf_spt[[1]] <- (sig / sqrt(1 - temp_corr^2)) * circulantEmbedding(obj)
for (i in seq_len(nt)[-1]) {
  rf_spt[[i]] <- temp_corr * rf_spt[[i - 1]] + sig * circulantEmbedding(obj)
}

coords <- as.matrix(expand.grid(x = grid[[1]], y = grid[[2]]))
N <- nrow(coords)

field <- sapply(rf_spt, as.vector)

beta0.true  <- 5
tau.sq.true <- 0.05
Y <- beta0.true + field + matrix(rnorm(N * nt, sd = sqrt(tau.sq.true)), N, nt)

sigma.sq.true <- sig^2
phi.true      <- 1 / range     ## fields' `range` <-> BRISC's phi = 1/range
nu.true       <- smoothness
gamma.true    <- temp_corr

cat(sprintf("N = %d sites, T = %d times (N*T = %d observations)\n\n", N, nt, N * nt))

##--- correlation function, matching BRISC's spCor() (src/util.cpp) ----

sp_cor <- function(D, phi, nu, cov.model = "matern") {
  switch(cov.model,
         exponential = exp(-phi * D),
         gaussian    = exp(-(phi * D)^2),
         spherical   = {
           r <- 1 - 1.5 * phi * D + 0.5 * (phi * D)^3
           r[D >= 1 / phi] <- 0
           r[D == 0] <- 1
           r
         },
         matern = {
           r <- (D * phi)^nu / (2^(nu - 1) * gamma(nu)) * besselK(D * phi, nu)
           r[D == 0] <- 1
           r
         },
         stop("unsupported cov.model"))
}

##--- dense reference: build Sigma = sigma.sq * (R_t %x% R_s) directly ----

dense_loglik_st <- function(coords, Y, beta, sigma.sq, tau.sq, phi, nu, gamma,
                            cov.model = "matern") {
  N <- nrow(Y)
  Tt <- ncol(Y)
  alpha <- tau.sq / sigma.sq
  Ds <- as.matrix(dist(coords))
  Rs <- sp_cor(Ds, phi, nu, cov.model) + alpha * diag(N)
  times <- seq_len(Tt)
  Dt <- abs(outer(times, times, "-"))
  Rt <- (gamma^Dt) / (1 - gamma^2)
  Sigma <- sigma.sq * kronecker(Rt, Rs)
  resid <- as.vector(Y - beta)
  R <- chol(Sigma)
  logdet <- 2 * sum(log(diag(R)))
  quad   <- sum(base::backsolve(R, resid, transpose = TRUE)^2)
  -0.5 * (N * Tt * log(2 * pi) + logdet + quad)
}

##--- fast path: NNGP (spatial) x exact AR(1) (temporal) Kronecker shortcuts ----
## Mirrors BRISC's own `updateBF` (spatial B/F) and the `likelihood_ST`
## profile log-likelihood in src/BRISC.cpp, using the exported
## `BRISC_neighbor()` for the (one-time) neighbor search.
fast_loglik_st <- function(coords, Y, beta, sigma.sq, tau.sq, phi, nu, gamma,
                            cov.model = "matern", n.neighbors) {
  N <- nrow(Y); Tt <- ncol(Y)
  alpha <- tau.sq / sigma.sq
  nb <- BRISC_neighbor(coords, n.neighbors = n.neighbors, ordering = seq_len(N),
                        verbose = FALSE)
  nnIndxLU <- matrix(nb$nnIndxLU, ncol = 2)
  nnIndx   <- nb$nnIndx
  Fvec  <- numeric(N)
  Blist <- vector("list", N)
  for (i in seq_len(N)) {
    ni <- nnIndxLU[i, 2]
    if (ni == 0) { Fvec[i] <- 1 + alpha; next }
    nn  <- nnIndxLU[i, 1]
    idx <- nnIndx[(nn + 1):(nn + ni)] + 1

    Dnn  <- as.matrix(dist(coords[idx, , drop = FALSE]))
    Cnn  <- sp_cor(Dnn, phi, nu, cov.model) + alpha * diag(ni)
    dvec <- sqrt(rowSums((coords[idx, , drop = FALSE] -
                             matrix(coords[i, ], ni, 2, byrow = TRUE))^2))
    cvec <- sp_cor(dvec, phi, nu, cov.model)

    b <- solve(Cnn, cvec)
    Blist[[i]] <- b
    Fvec[i] <- 1 + alpha - sum(b * cvec)
  }
  logDetF  <- sum(log(Fvec))
  logDetFt <- log(1 - gamma^2)
  whiten_col <- function(v) {
    out <- numeric(N)
    out[1] <- v[1] / sqrt(Fvec[1])
    for (i in 2:N) {
      ni <- nnIndxLU[i, 2]
      if (ni == 0) { out[i] <- v[i] / sqrt(Fvec[i]); next }
      nn <- nnIndxLU[i, 1]
      idx <- nnIndx[(nn + 1):(nn + ni)] + 1
      out[i] <- (v[i] - sum(Blist[[i]] * v[idx])) / sqrt(Fvec[i])
    }
    out
  }
  E  <- Y - beta
  Zs <- apply(E, 2, whiten_col)
  W  <- matrix(0, N, Tt)
  W[, 1] <- sqrt(1 - gamma^2) * Zs[, 1]
  if (Tt > 1) for (t in 2:Tt) W[, t] <- Zs[, t] - gamma * Zs[, t - 1]

  sse    <- sum(W^2)
  logdet <- N * Tt * log(sigma.sq) - N * logDetFt + Tt * logDetF

  -0.5 * (N * Tt * log(2 * pi) + logdet + sse / sigma.sq)
}

##--- compare accuracy and timing, evaluated at the true parameters ----

cat("---- dense (explicit N*T x N*T Kronecker product) ----\n")
t_dense <- system.time(
  llk_dense <- dense_loglik_st(coords, Y, beta0.true, sigma.sq.true, tau.sq.true,
                                phi.true, nu.true, gamma.true)
)
cat(sprintf("log-lik = %.6f, elapsed = %.4fs\n\n", llk_dense, t_dense["elapsed"]))

cat("---- fast (NNGP spatial decorrelation x exact AR(1)) ----\n")
for (m in c(5, 10, 15, N - 1)) {
  t_fast <- system.time(
    llk_fast <- fast_loglik_st(coords, Y, beta0.true, sigma.sq.true, tau.sq.true,
                                phi.true, nu.true, gamma.true, n.neighbors = m)
  )
  cat(sprintf("n.neighbors = %3d:  log-lik = %.6f  (diff from dense = %.3e),  elapsed = %.4fs\n",
              m, llk_fast, llk_dense - llk_fast, t_fast["elapsed"]))
}

cat(sprintf("\nSpeed-up (dense / fast, n.neighbors = 15): %.1fx\n",
            t_dense["elapsed"] /
              system.time(fast_loglik_st(coords, Y, beta0.true, sigma.sq.true, tau.sq.true,
                                          phi.true, nu.true, gamma.true,
                                          n.neighbors = 15))["elapsed"]))
