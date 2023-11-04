library(sf)
library(spatstat)

path <- "C:/Users/lisak/OneDrive/PhD2.0/Coding/Simulations/spatstat/final_ex1/"
load(paste0(path, "knp_poach.RData"))

## windows

# calculate window from projected grid and sample window
w <- as.owin(grid_proj) # full window
w <- rescale(w, 1000, "km")
b <- owin(c(-68.33145, -35.92676), c(2694.3555, 2715.5711))
b <- w[b] # smaller sampling window
notb <- setminus.owin(w, b) # complement of b in w
w_area <- area.owin(w)
b_area <- area.owin(b)
paste0("|W| = : ", round(w_area, 2), " km^2, |B| = ", round(b_area, 2), " km^2")

# plot windows
xlab <- "Easting distance (km)"
ylab <- "Northing distance (km)"
pdf(paste0(path, "knp_window.pdf"), height = 2, width = 7)
par(mfrow = c(1, 3), mar = c(1, 2, 1, 1), oma = c(3, 3, 0, 0))
plot(reflect(w), main = "", axes = TRUE)
plot(reflect(w), main = "", axes = TRUE)
plot(reflect(b), add = TRUE, hatch = TRUE)
plot(reflect(b), main = "", axes = TRUE)
mtext(xlab, side = 1, line = 1.5, cex = 0.8, outer = TRUE)
mtext(ylab, side = 2, line = 1, cex = 0.8, outer = TRUE)
dev.off()

# calculate pixel images in sample window
poach_sub <- poach_prob[b, drop = FALSE, tight = TRUE]
poach_sub <- poach_sub / sum(poach_sub) # normalize
image_subset <- function(img) {
  img[b, drop = FALSE, tight = TRUE]
}
covar_sub <- lapply(covariates, image_subset)
all_images_sub <- as.solist(c(list(poach = poach_sub), covar_sub))

# plot pixel images in sample window
panel <- c("(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)")
pdf(paste0(path, "cov_window.pdf"), height = 9, width = 7)
par(mfrow = c(4, 2), mar = c(2, 2, 1, 3), cex.main = 0.8, oma = c(3, 3, 0, 0))
for (i in seq_along(all_images_sub)) {
  rlab <- ifelse(i == 1, "Probability", "Distance (km)")
  plot(reflect(all_images_sub[[i]]), main = "", axes = TRUE,  ribsep = 0.05)
  mtext(rlab, side = 4, line = 1.75, cex = 0.8)
  mtext(bquote(italic(.(panel[i]))), side = 3, line = -0.5, cex = 0.8)
  if (i == 1) {
    mtext(xlab, side = 1, line = 0.5, cex = 0.8, outer = TRUE)
    mtext(ylab, side = 2, line = 1, cex = 0.8, outer = TRUE)
  }
}
dev.off()

# Sample window of Poisson
set.seed(12345)
mu <- 200 # expected number of points
p <- 0.8 * b_area / w_area
n_large <- mu / p
lambda_large <- poach_prob * round(n_large)
pp_large <- rpoispp(lambda_large)
pp_win <- pp_large[b]
pp_comp <- pp_large[notb]

# Quadrat tests for CSR
pp_all <- solist(pp_large, pp_win)
pp_names <- c("Inhomogeneous Poisson", "Sample window")
names(pp_all) <- pp_names
qtest_x2 <- lapply(pp_all, function(x) quadrat.test(x, nx = 4, ny = 3))
qtest_x2_stat <- sapply(qtest_x2, function(x) x$statistic)
qtest_x2_pval <- sapply(qtest_x2, function(x) x$p.value)
qtest_x2_df <- sapply(qtest_x2, function(x) x$parameter)
qtest <- data.frame(X2 = qtest_x2_stat, df = qtest_x2_df, pval = qtest_x2_pval)
rownames(qtest) <- pp_names
qtest <- round(qtest, 3)

# plot lambda and inhomogeneous Poisson
rlab <- bquote("Intensity " ~ lambda(u))
pdf(paste0(path, "knp_poisson_pp.pdf"), height = 4, width = 6)
par(mfrow = c(1, 1), mar = c(3, 2, 1, 3))
plot(reflect(lambda_large), main = "", axes = TRUE,
  ribsep = 0.05, cex.axis = 0.7,
        col = hcl.colors(128, alpha = 0.8, palette = "Grays"))
plot(reflect(pp_large), show.window = FALSE,
  add = TRUE, pch = 16, cex = 0.8,
        cols = rgb(0, 0, 0, 0.5))
mtext(rlab, side = 4, line = 1.75, cex = 0.8)
mtext(xlab, side = 1, line = 1.5, cex = 0.8)
mtext(ylab, side = 2, line = 1, cex = 0.8)
dev.off()

# plot window sampling
pdf(paste0(path, "knp_window_pp.pdf"), height = 3, width = 7)
par(mfrow = c(1, 2), mar = c(1, 2, 1, 1), oma = c(2, 2, 0, 0))
plot(reflect(w), main = "", axes = TRUE)
plot(reflect(pp_comp), show.window = FALSE, add = TRUE, pch = 16, cex = 0.5)
plot(reflect(b), add = TRUE)
plot(reflect(pp_win), show.window = FALSE, add = TRUE, pch = 16, cex = 0.5,
      col = "tomato")
plot(reflect(b), main = "", axes = TRUE, cex.axis = 0.6)
plot(reflect(pp_win), show.window = FALSE, add = TRUE, pch = 16, cex = 0.5,
      col = "tomato")
mtext(xlab, side = 1, line = 1, cex = 1, outer = TRUE)
mtext(ylab, side = 2, line = 1, cex = 1, outer = TRUE)
dev.off()

# print results
mu
p
n_large
summary(lambda_large)
summary(pp_large)
summary(pp_win)
qtest

rm(i, panel, qtest_x2_df, qtest_x2_pval, qtest_x2_stat, rlab, xlab, ylab)
save.image(file = paste0(path, "knp_window.RData"))