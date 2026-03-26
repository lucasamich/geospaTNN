library(fields)

##--- spatial only ----
nx <- 50
ny <- 50
cov_model <- "Matern"
range <- .5
smoothness <- 1
sig <- 1
grid <- list(x = seq(0, 5,
                     len = nx),
             y = seq(0, 5,
                     len = ny)) 
obj <- circulantEmbeddingSetup(grid,
                               Covariance = cov_model,
                               aRange = range,
                               smoothness = smoothness)

set.seed(223)

rf1 <- sig * circulantEmbedding(obj)

rf2 <- sig * circulantEmbedding(obj)

set.panel(1, 2)
image.plot(grid[[1]], grid[[2]], rf1)
title("simulated gaussian fields")
image.plot( grid[[1]], grid[[2]], rf2)
title("another realization ...")

##--- spatiotemporal ----

nt <- 20
## AR(1) parameter
temp_corr <- .6

rf_spt <- vector(mode = "list", length = nt)

rf_spt[[1]] <- rf1


for (i in seq_len(nt)[-1]) {
  rf_spt[[i]] <- temp_corr * rf_spt[[i - 1]] +
    sig * circulantEmbedding(obj)
}

## viz a few timepoints

set.panel(2, 2)
for (t in c(1, 2, 19, 20)) {
  image.plot(grid[[1]], grid[[2]],
             rf_spt[[t]],
             xlab = "x", ylab = "y")
  title(sprintf("t = %d", t))
}

output_spt <- do.call(rbind, rf_spt)
space <- rep(seq_len(length(rf_spt[[1]])),
             rep = length(rf_spt))
time <- sapply(seq_along(rf_spt),
               \(x) rep(x, length(rf_spt[[1]]))) |>
  c()
