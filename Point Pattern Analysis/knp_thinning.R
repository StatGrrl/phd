library(sf)
library(spatstat)

path <- "C:/Users/lisak/OneDrive/PhD2.0/Coding/Simulations/spatstat/final_ex1/"
load(paste0(path, "knp_poach.RData"))

# get window
w <- as.owin(grid_proj) # full window
w <- rescale(w, 1000, "km")
w_area <- area.owin(w)

set.seed(1000000)
mu <- 200 # expected number of points
p <- 0.5

# p thinning
n_p <- round(mu / p)
lambda_p <- poach_prob * n_p
pp_large_p <- rpoispp(lambda_p) # inhomogeneous Poisson
pp_thin_p <- rthin(pp_large_p, p)
pp_tmp <- superimpose(pp_thin_p, pp_large_p)
pp_comp_p <- pp_tmp[-which(multiplicity(pp_tmp) > 1)]

# p(x) thinning
p_px <- poach_prob * p /  mean(poach_prob)
p_px <- p_px / max(p_px)
n_px <- round(mu / mean(p_px))
lambda_px <- n_px / w_area
pp_large_px <- rpoispp(lambda_px, win = w) # homogeneous Poisson
pp_thin_px <- rthin(pp_large_px, p_px)
pp_tmp <- superimpose(pp_thin_px, pp_large_px)
pp_comp_px <- pp_tmp[-which(multiplicity(pp_tmp) > 1)]

# P(X) thinning
p_rpx <- rnoise(runif, poach_prob, max = 0.5)
p_rpx <- Smooth(p_rpx, sigma = 1, normalise = TRUE, bleed = FALSE)
p_rpx <- p_rpx + 0.5 * (poach_prob * p /  mean(poach_prob))
n_rpx <- round(mu / mean(p_rpx))
lambda_rpx <- n_rpx / w_area
pp_large_rpx <- rpoispp(lambda_rpx, win = w) # homogeneous Poisson
pp_thin_rpx <- rthin(pp_large_rpx, p_rpx)
pp_tmp <- superimpose(pp_thin_rpx, pp_large_rpx)
pp_comp_rpx <- pp_tmp[-which(multiplicity(pp_tmp) > 1)]

# Quadrat tests for CSR
pp_all <- solist(pp_large_p, pp_thin_p, pp_large_px, pp_thin_px,
            pp_large_rpx, pp_thin_rpx)
pp_names <- c("Inhomogeneous Poisson", "p-Thinning", "Homogeneous Poisson 1",
            "p(x)-Thinning", "Homogeneous Poisson 2", "P(X)-Thinning")
param_all <- anylist(lambda_p, p, lambda_px, p_px, lambda_rpx, p_rpx)
names(pp_all) <- pp_names
qtest_x2 <- lapply(pp_all, function(x) quadrat.test(x, nx = 4, ny = 3))
qtest <- data.frame(param_min = sapply(param_all, min),
                    param_max = sapply(param_all, max),
                    param_mean = sapply(param_all, mean),
                    npoints = sapply(pp_all, npoints),
                    intensity = sapply(pp_all, intensity),
                    X2 = sapply(qtest_x2, function(x) x$statistic),
                    df = sapply(qtest_x2, function(x) x$parameter),
                    pval = sapply(qtest_x2, function(x) x$p.value))
rownames(qtest) <- pp_names
qtest <- round(qtest, 3)
qtest

# plot thinned process

# p-thinning of inhomogeneous Poisson
pdf(paste0(path, "knp_thinning_p.pdf"), height = 2, width = 7)
par(mfrow = c(1, 3), mar = c(0, 0, 0, 0), oma = c(1, 1, 1, 1), xpd = NA)
plot(reflect(w), main = "")
plot(reflect(pp_large_p), show.window = FALSE, add = TRUE, pch = 16,
      cex = 0.7, col = "tomato", alpha = 0.8)
plot(reflect(w), main = "")
plot(reflect(pp_thin_p), show.window = FALSE, add = TRUE, pch = 16,
      cex = 0.7, col = "tomato", alpha = 0.8)
plot(reflect(pp_comp_p), show.window = FALSE, add = TRUE, pch = 16,
      cex = 0.7, alpha = 0.8)
plot(reflect(w), main = "")
plot(reflect(pp_thin_p), show.window = FALSE, add = TRUE, pch = 16,
      cex = 0.7, col = "tomato", alpha = 0.8)
box("inner")
dev.off()

# p(x)-thinning of homogeneous poisson
pdf(paste0(path, "knp_thinning_px.pdf"), height = 2, width = 7)
par(mfrow = c(1, 3), mar = c(0, 0, 0, 0), oma = c(1, 1, 1, 1), xpd = NA)
plot(reflect(w), main = "")
plot(reflect(pp_large_px), show.window = FALSE, add = TRUE, pch = 16,
      cex = 0.7, col = "steelblue", alpha = 0.8)
plot(reflect(w), main = "")
plot(reflect(pp_thin_px), show.window = FALSE, add = TRUE, pch = 16,
      cex = 0.7, col = "tomato", alpha = 0.8)
plot(reflect(pp_comp_px), show.window = FALSE, add = TRUE, pch = 16,
      cex = 0.7, alpha = 0.8)
plot(reflect(w), main = "")
plot(reflect(pp_thin_px), show.window = FALSE, add = TRUE, pch = 16,
      cex = 0.7, col = "tomato", alpha = 0.8)
box("inner")
dev.off()

# P(x)-thinning of inhomogeneous poisson
pdf(paste0(path, "knp_thinning_rpx.pdf"), height = 2, width = 7)
par(mfrow = c(1, 3), mar = c(0, 0, 0, 0), oma = c(1, 1, 1, 1), xpd = NA)
plot(reflect(w), main = "")
plot(reflect(pp_large_rpx), show.window = FALSE, add = TRUE, pch = 16,
      cex = 0.7, col = "steelblue", alpha = 0.8)
plot(reflect(w), main = "")
plot(reflect(pp_thin_rpx), show.window = FALSE, add = TRUE, pch = 16,
      cex = 0.7, col = "goldenrod", alpha = 0.8)
plot(reflect(pp_comp_rpx), show.window = FALSE, add = TRUE, pch = 16,
      cex = 0.7, alpha = 0.8)
plot(reflect(w), main = "")
plot(reflect(pp_thin_rpx), show.window = FALSE, add = TRUE, pch = 16,
      cex = 0.7, col = "goldenrod", alpha = 0.8)
box("inner")
dev.off()

# plot retention probability images
xlab <- "Easting distance (km)"
ylab <- "Northing distance (km)"
rlab <- "Retention probability"
pdf(paste0(path, "knp_thinning_prob.pdf"), height = 4, width = 7)
par(mfrow = c(1, 2), mar = c(2, 2, 2, 0), oma = c(1, 1, 1, 0))
plot(reflect(p_px), main = "", axes = TRUE, cex.axis = 0.7, ribsep = 0.05,
    ribside = "top")
mtext(rlab, side = 3, line = 1, cex = 0.8)
plot(reflect(p_rpx), main = "", axes = TRUE, cex.axis = 0.7, ribsep = 0.05,
    ribside = "top")
mtext(rlab, side = 3, line = 1, cex = 0.8)
mtext(xlab, side = 1, line = -0.5, cex = 0.8, outer = TRUE)
mtext(ylab, side = 2, line = 0, cex = 0.8, outer = TRUE)
dev.off()

save.image(file = paste0(path, "knp_thinning.RData"))