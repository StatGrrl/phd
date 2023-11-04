library(spatstat)

path <- "C:/Users/lisak/OneDrive/PhD2.0/Coding/Simulations/spatstat/final_ex1/"

# binomial and poisson pp in unit square
set.seed(1)
pdf(paste0(path, "ex_pp.pdf"), height = 4, width = 7)
par(mfrow = c(3, 5), mar = c(0.5, 0, 0.5, 0))
for (i in 1:5) {
  plot(runifpoint(10), main  = "", pch = 16, cols = "mediumseagreen")
}
for (i in 1:5) {
  plot(rpoispp(10), main  = "", pch = 16, cols = "steelblue")
}
for (i in 1:5) {
  plot(rpoispp(function(x, y) 10 * (x + y)), main  = "", pch = 16,
    cols = "tomato")
}
dev.off()

# csr, regular and cluster pp in unit square
set.seed(2)
pdf(paste0(path, "ex_interaction.pdf"), height = 4, width = 7)
par(mfrow = c(3, 5), mar = c(0.5, 0, 0.5, 0))
for (i in 1:5) { # CSR
  plot(rpoispp(lambda = 20), main  = "", pch = 16, cols = "steelblue")
}
for (i in 1:5) { # simple sequential inhibition
  plot(rSSI(r = 0.1, n = 20), main  = "", pch = 16, cols = "plum")
}
for (i in 1:5) { # Matern cluster
  plot(rMatClust(kappa = 5, scale = 0.1, mu = 4), main  = "", pch = 16,
    cols = "lightsalmon")
}
dev.off()

# csr, mixed poisson pp in unit square
set.seed(3)
lambda_random <- vector()
pdf(paste0(path, "ex_mixed_pois.pdf"), height = 2.5, width = 7)
par(mfrow = c(2, 5), mar = c(0.5, 0, 0.5, 0))
for (i in 1:5) { # CSR
  plot(rpoispp(lambda = 10), main  = "", pch = 16, cols = "steelblue")
}
for (i in 1:5) { # mixed poisson
  lambda <- rexp(1, rate = 1 / 10)
  lambda_random <- c(lambda_random, lambda)
  plot(rpoispp(lambda), main  = "", pch = 16, cols = "goldenrod")
}
dev.off()
round(lambda_random, 2)

# grid of points
pp <- rpoispp(lambda = 20)
quad <- quadratcount(pp, nx = 10, ny = 10)
pdf(paste0(path, "binary_grid.pdf"), width = 3, height = 3)
par(mar = c(0, 0, 0, 0))
plot(quad, col = rep("gray", 100), do.col = TRUE, ribbon = FALSE, main = "")
plot(pp, pch = 16, cols = "white", add = TRUE, cex = 1.5)
plot(quad, add = TRUE, entries = rep("", 100))
plot(quad, show.tiles = FALSE, add = TRUE)
dev.off()
