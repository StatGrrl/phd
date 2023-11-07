library(sf)
library(spatstat)
library(ggplot2)
theme_set(theme_bw())
library(patchwork)

path <- "C:/Users/lisak/OneDrive/PhD2.0/Coding/Simulations/spatstat/final_ex1/"
load(paste0(path, "knp_poach.RData"))

# get window
w <- as.owin(grid_proj) # full window
w <- rescale(w, 1000, "km")

# simulate data
set.seed(10)
mu <- 200
p <- 0.8
n_large <- round(mu / p)
lambda <- poach_prob * n_large
pp <- rpoispp(lambda)
sample <- rthin(pp, p)

# plot point pattern and quadrat test
n <- npoints(sample)
lambda_bar <- intensity(sample)
qc <- quadratcount(sample, nx = 4, ny = 3)
qt <- quadrat.test(qc)

xlab <- "Westing distance (km)"
ylab <- "Southing distance (km)"
pdf(paste0(path, "explore_csr_test.pdf"), width = 5.5, height = 4)
par(mfrow = c(1, 1), mar = c(3, 1, 1, 0))
plot(qc, main = "", axes = TRUE, cex.axis = 0.8, ribbon = FALSE,
    textargs = list(adj = c(0, -1.5)))
plot(sample, add = TRUE, pch = 16, cols = rgb(0, 0, 0, 0.2))
mtext(xlab, side = 1, line = 2)
mtext(ylab, side = 2, line = 0)
dev.off()

summary(sample)
qt

# plot density
xlab <- "Easting distance (km)"
ylab <- "Northing distance (km)"
pdf(paste0(path, "explore_density.pdf"), width = 7, height = 4)
par(mfrow = c(1, 2), mar = c(2, 2, 2, 0), oma = c(1, 1, 1, 0))
plot(reflect(lambda), main = "", axes = TRUE, cex.axis = 0.8,
    ribsep = 0.05, ribside = "top")
plot(reflect(w), main = "", add = TRUE)
plot(reflect(pp), show.window = FALSE, add = TRUE, pch = 16,
      cex = 0.7, alpha = 0.5)
mtext("Intensity", side = 3, line = 1, cex = 0.8)
plot(reflect(density(sample)), main = "", axes = TRUE,
    cex.axis = 0.8, ribsep = 0.05, ribside = "top")
plot(reflect(w), main = "", add = TRUE)
plot(reflect(sample), show.window = FALSE, add = TRUE, pch = 16,
      cex = 0.7, alpha = 0.5)
mtext("Density Estimate", side = 3, line = 1, cex = 0.8)
mtext(xlab, side = 1, line = -1, outer = TRUE)
mtext(ylab, side = 2, line = 0, outer = TRUE)
dev.off()

# plot histograms of images
xlab <- "Distance (km)"
ylab <- "Frequency"
panel <- c("(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)")
pdf(paste(path, "explore_cov_hist.pdf", sep = ""), height = 9, width = 7)
par(mfrow = c(4, 2), mar = c(4, 2, 3, 1), cex.main = 1, oma = c(3, 3, 0, 0))
for (i in seq_along(covariates)) {
  hist(covariates[[i]], main = panel[i], xlab = xlab,
    ylab = "", breaks = "Scott")
}
hist(poach_prob, main = "(h)", xlab = "Probability",
    ylab = "", breaks = "Scott")
mtext(ylab, side = 2, line = 1, cex = 0.8, outer = TRUE)
dev.off()

# plots and table of quadrats for covariates (first plot point pattern)
cov_quad <- data.frame()
fit_uni <- ppm(sample ~ 1)
panel <- c("(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)")
cov_dir <- c(TRUE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE)
pdf(paste(path, "explore_cov_quadrats.pdf", sep = ""), height = 7, width = 6)
par(mfrow = c(4, 2), mar = c(0, 0, 1, 2), cex.main = 0.8, oma = c(0, 0, 0, 0))
for (i in seq_along(covariates)) {
  z <- covariates[[i]]
  v <- quantess(sample, z, 4)
  qb <- quadratcount(sample, tess = v)
  plot(qb, main = "", ribsep = 0.05)
  mtext(bquote(italic(.(panel[i]))), side = 3, line = -0.25, cex = 0.8)
  cov_ppm <- ppm(sample ~ ., covariates = covariates[i])
  cov_entry <- data.frame(Q1 = qb[1], Q2 = qb[2], Q3 = qb[3], Q4 = qb[4],
    Chi2 = quadrat.test(qb, alternative = "two.sided")$p.value,
    Berman_Z1 = berman.test(pp, z)$p.value,
    Berman_Z2 = berman.test(pp, z, "Z2")$p.value,
    KS = cdf.test(sample, z, test = "ks")$p.value,
    LRT = anova(fit_uni, cov_ppm, test = "LRT")$"Pr(>Chi)"[2],
    AUC = auc(sample, z, high = cov_dir[i]),
    row.names = names(covariates)[i])
  cov_quad <- rbind(cov_quad, cov_entry)
}
plot(lambda, main = "", ribsep = 0.05, axes = TRUE)
plot(sample, add = TRUE, pch = 16)
mtext(bquote(italic(.(panel[i]))), side = 3, line = -0.25, cex = 0.8)
dev.off()
round(cov_quad, 3)

# covariate quadrat barplots
xlab <- "Distance Quantiles (km)"
ylab <- "Number of Points"
panel <- c("(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)")
pdf(paste0(path, "explore_cov_quadrats_barplots.pdf"), height = 3, width = 6)
par(mfrow = c(2, 4), mar = c(2, 2, 2, 0), oma = c(3, 3, 0, 0), cex.main = 1)
for (i in seq_along(covariates)) {
  z <- covariates[[i]]
  v <- quantess(sample, z, 4)
  qb <- quadratcount(sample, tess = v)
  barplot(qb, main = panel[i], names.arg = c("Q1", "Q2", "Q3", "Q4"))
}
mtext(xlab, side = 1, line = 0.5, cex = 0.8, outer = TRUE)
mtext(ylab, side = 2, line = 1, cex = 0.8, outer = TRUE)
dev.off()

# intensity estimate rho(cov)
xlab <- "Distance (km)"
ylab <- "Intensity Estimate"
panel <- c("(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)")
pdf(paste(path, "explore_cov_rhohat.pdf", sep = ""), height = 9, width = 7)
par(mfrow = c(4, 2), mar = c(2, 2, 1, 1), cex.main = 1, oma = c(3, 3, 0, 0))
for (i in seq_along(covariates)) {
  plot(rhohat(sample, covariates[[i]]), main = "", legend=FALSE)
}
mtext(xlab, side = 1, line = 0.5, cex = 0.8, outer = TRUE)
mtext(ylab, side = 2, line = 1, cex = 0.8, outer = TRUE)
dev.off()

# intensity estimate rho(cov)
pdf(paste(path, "explore_cov_rhohat_predict.pdf", sep = ""),
    height = 9, width = 7)
par(mfrow = c(4, 2), mar = c(2, 2, 2, 1), cex.main = 0.8, oma = c(3, 3, 0, 0))
for (i in seq_along(covariates)) {
  plot(reflect(predict(rhohat(sample, covariates[[i]]))), main = panel[i],
    axes = TRUE, cex.axis = 0.8, ribsep = 0.05, ribside = "top")
}
mtext(xlab, side = 1, line = 0.5, cex = 0.8, outer = TRUE)
mtext(ylab, side = 2, line = 1, cex = 0.8, outer = TRUE)
dev.off()

# cdf kolmogorov-smirnov test
xlab <- "Distance (km)"
ylab <- "Probability"
panel <- c("(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)")
pdf(paste(path, "explore_cov_ks.pdf", sep = ""), height = 9, width = 7)
par(mfrow = c(4, 2), mar = c(2, 2, 1, 1), cex.main = 1, oma = c(3, 3, 0, 0))
for (i in seq_along(covariates)) {
  plot(cdf.test(sample, covariates[[i]], test = "ks"), main = "",
    xlab = "", ylab = "")
}
mtext(xlab, side = 1, line = 0.5, cex = 0.8, outer = TRUE)
mtext(ylab, side = 2, line = 1, cex = 0.8, outer = TRUE)
dev.off()

# cdf PP plot
xlab <- "Theoretical Probability"
ylab <- "Observed Probability"
panel <- c("(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)")
pdf(paste(path, "explore_cov_pp.pdf", sep = ""), height = 9, width = 7)
par(mfrow = c(4, 2), mar = c(2, 2, 2, 1), cex.main = 1, oma = c(3, 3, 0, 0))
for (i in seq_along(covariates)) {
  plot(cdf.test(sample, covariates[[i]], test = "ks"), style = "PP",
    main = panel[i])
#    xlab = "", ylab = "")
}
mtext(xlab, side = 1, line = 0.5, cex = 0.8, outer = TRUE)
mtext(ylab, side = 2, line = 1, cex = 0.8, outer = TRUE)
dev.off()

# cdf QQ plot
xlab <- "Theoretical quantile of covariate"
ylab <- "Observed quantile of covariate"
panel <- c("(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)")
pdf(paste(path, "explore_cov_qq.pdf", sep = ""), height = 9, width = 7)
par(mfrow = c(4, 2), mar = c(2, 2, 2, 1), cex.main = 1, oma = c(3, 3, 0, 0))
for (i in seq_along(covariates)) {
  plot(cdf.test(sample, covariates[[i]], test = "ks"), style = "QQ",
    main = panel[i],
    xlab = "", ylab = "")
}
mtext(xlab, side = 1, line = 0.5, cex = 0.8, outer = TRUE)
mtext(ylab, side = 2, line = 1, cex = 0.8, outer = TRUE)
dev.off()

# berman Z1 test
xlab <- "Distance (km)"
ylab <- "Probability"
panel <- c("(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)")
pdf(paste(path, "explore_cov_z1.pdf", sep = ""), height = 9, width = 7)
par(mfrow = c(4, 2), mar = c(2, 2, 1, 1), cex.main = 1, oma = c(3, 3, 0, 0))
for (i in seq_along(covariates)) {
  plot(berman.test(sample, covariates[[i]], "Z1"), main = "",
    xlab = "", ylab = "")
}
mtext(xlab, side = 1, line = 0.5, cex = 0.8, outer = TRUE)
mtext(ylab, side = 2, line = 1, cex = 0.8, outer = TRUE)
dev.off()

# berman Z2 test
xlab <- "Spatial cdf"
ylab <- "Relative Frequency"
panel <- c("(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)")
pdf(paste(path, "explore_cov_z2.pdf", sep = ""), height = 9, width = 7)
par(mfrow = c(4, 2), mar = c(2, 2, 2, 1), cex.main = 1, oma = c(3, 3, 0, 0))
for (i in seq_along(covariates)) {
  plot(berman.test(sample, covariates[[i]], "Z2"), main = panel[i],
    xlab = "", ylab = "")
}
mtext(xlab, side = 1, line = 0.5, cex = 0.8, outer = TRUE)
mtext(ylab, side = 2, line = 1, cex = 0.8, outer = TRUE)
dev.off()

# roc curves
xlab <- "Distance between points"
ylab <- "Fraction of points "
panel <- c("(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)")
cov_dir <- c(TRUE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE)
pdf(paste(path, "explore_cov_roc.pdf", sep = ""), height = 9, width = 7)
par(mfrow = c(4, 2), mar = c(2, 2, 1, 1), cex.main = 1, oma = c(3, 3, 0, 0))
for (i in seq_along(covariates)) {
  plot(roc(sample, covariates[[i]], high = cov_dir[i]), main = "",
    xlab = "", ylab = "", legend=FALSE)
}
mtext(xlab, side = 1, line = 0.5, cex = 0.8, outer = TRUE)
mtext(ylab, side = 2, line = 1, cex = 0.8, outer = TRUE)
dev.off()

save.image(file = paste0(path, "knp_pp_explore.RData"))