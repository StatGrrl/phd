library(spatstat)
library(raster)
library(sf)
library(maptools)
library(rgdal)
library(ggplot2)
theme_set(theme_bw())
library(patchwork)

### point pattern analysis - kruger example 1

## get covariate data and poach probability

# read polygon grids and rasterize using projected grid
path <- "C:/Users/lisak/OneDrive/PhD2.0/Coding/Simulations/spatstat/final_ex1/"
grid_gps <- st_read(paste0(path, "shapefiles/make_grid_gps.shp"))
grid_proj <- st_read(paste0(path, "shapefiles/make_grid_proj.shp"))
s <- as(grid_proj, "Spatial")
r <- raster(s, ncols = 62, nrows = 44)
poach_prob  <- rasterize(s, r, "Select.Nor", background = NA)
dist_roads  <- rasterize(s, r, "Dist.Roads", background = NA)
dist_camps  <- rasterize(s, r, "Dist.Camps", background = NA)
dist_picnic  <- rasterize(s, r, "Dist.Picni", background = NA)
dist_gates  <- rasterize(s, r, "Dist.Gates", background = NA)
dist_border  <- rasterize(s, r, "Dist.Borde", background = NA)
dist_dams  <- rasterize(s, r, "Dist.Dams", background = NA)
dist_water  <- rasterize(s, r, "Dist.Water", background = NA)

# make poacher probability and covariate pixel images rescaled to km
poach_prob <- as.im.RasterLayer(poach_prob)
poach_prob <- rescale(poach_prob, s = 1000, unitname = "km")
spat_cov <- list(dist_roads, dist_camps, dist_picnic, dist_gates,
            dist_border, dist_dams, dist_water)
names <- c("roads", "camps", "picnic", "gates", "border", "dams", "water")
descrip <- c("Distance to nearest road (km)", "Distance to nearest camp (km)",
            "Distance to nearest picnic spot (km)",
            "Distance to nearest gate (km)",
            "Distance to nearest border (km)", "Distance to nearest dam (km)",
            "Distance to nearest water (km)")
m_to_km <- function(x) x / 1000
spat_cov <- lapply(spat_cov, m_to_km)
spat_cov <- lapply(spat_cov, as.im.RasterLayer)
spat_cov <- setNames(spat_cov, names)
spat_cov <- lapply(spat_cov, rescale, s = 1000, unitname = "km")
all_images <- as.solist(c(list(poach = poach_prob), spat_cov))
covariates <- as.solist(spat_cov)

# plot pixel images
pdf(paste0(path, "image_colourmap.pdf"), height = 9, width = 7)
par(mfrow = c(4, 2), mar = c(1, 0, 1, 1) + 0.1, cex.main = 0.8)
for (i in seq_along(all_images)) {
  plot(all_images[[i]], main = c("Poacher probability", descrip)[i],
        ribsep = 0.05)
}
dev.off()

# plot histograms of images
pdf(paste0(path, "image_histogram.pdf"), height = 9, width = 7)
par(mfrow = c(4, 2), mar = c(5, 5, 2, 2), cex.main = 0.8)
for (i in seq_along(all_images)) {
  hist(all_images[[i]], main = c("Poacher probability", descrip)[i],
      xlab = c("Probability", rep("Distance (km)", length(covariates)))[i],
      ylab = "Frequency", breaks = "Scott")
}
dev.off()

## windows

# calculate window from projected grid and sample window
w <- as.owin(grid_proj)
w <- rescale(w, 1000, "km")
b <- owin(c(-68.33145, -35.92676), c(2694.3555, 2715.5711))
b <- w[b] # smaller sampling window
w_area <- area.owin(w)
b_area <- area.owin(b)
notb <- setminus.owin(w, b) # complement of b in w

# plot windows
pdf(paste0(path, "windows.pdf"), height = 2, width = 7)
par(mfrow = c(1, 3), mar = c(1, 0, 1, 1) + 0.1, cex.main = 0.8)
plot(w, main = paste0("Full window W, area = ", round(w_area, 2), " square km"))
plot(w, main = "Location of sample window")
plot(b, add = TRUE, hatch = TRUE)
plot(b, main = paste0("Sample window B, area = ", round(b_area, 2),
  " square km"))
dev.off()

# calculate pixel images in sample window
poach_sub <- poach_prob[b, drop = FALSE, tight = TRUE]
poach_sub <- poach_sub / sum(poach_sub) # normalize
image_subset <- function(img) {
  img[b, drop = FALSE, tight = TRUE]
}
spat_cov_sub <- lapply(covariates, image_subset)
all_images_sub <- as.solist(c(list(poach = poach_sub), spat_cov_sub))

# plot pixel images in sample window
pdf(paste0(path, "image_colour_sub.pdf"), height = 9, width = 7)
par(mfrow = c(4, 2), mar = c(1, 0, 1, 1) + 0.1, cex.main = 0.8)
for (i in seq_along(all_images_sub)) {
  plot(all_images_sub[[i]], main = c("Poacher probability", descrip)[i],
        ribsep = 0.05)
}
dev.off()

# plot histograms of images in sample window
pdf(paste0(path, "image_hist_sub.pdf"), height = 9, width = 7)
par(mfrow = c(4, 2), mar = c(5, 5, 2, 2), cex.main = 0.8)
for (i in seq_along(all_images_sub)) {
  hist(all_images_sub[[i]], main = c("Poacher probability", descrip)[i],
      xlab = c("Probability", rep("Distance (km)", length(covariates)))[i],
      ylab = "Frequency", breaks = "Scott")
}
dev.off()

## generating point patterns

# simple example
pdf(paste0(path, "ex_pp.pdf"), height = 5.5, width = 7)
par(mfrow = c(3, 5), mar = c(0, 0, 1, 0), cex.main = 0.9, oma = c(1, 1, 1, 1))
for (i in 1:5) {
  title <- if (i == 3) "Binomial" else ""
  plot(runifpoint(10), main  = title)
}
for (i in 1:5) {
  title <- if (i == 3) "Homogeneous Poisson" else ""
  plot(rpoispp(10), main  = title )
}
for (i in 1:5) {
  title <- if (i == 3) "Inhomogeneous Poisson" else ""
  plot(rpoispp(function(x, y) { 10 * (x + y)}), main  = title)
}
dev.off()

set.seed(12345)
exp_n <- 200 # number of points

# CSR
lambda_csr <- exp_n / w_area
pp_csr <- rpoispp(lambda_csr, win = w)

# Poisson
lambda_pois <- poach_prob * exp_n
pp_pois <- rpoispp(lambda_pois)

# Sample window of Poisson
p_win <- 0.8 * b_area / w_area
n_large_win <- exp_n / p_win
lambda_large_win <- poach_prob * n_large_win
pp_large_win <- rpoispp(lambda_large_win)
pp_win <- pp_large_win[b]
pp_comp_win <- pp_large_win[notb]

# Thinned Poisson
p_max <- 0.5
# p-thinning
p_thin1 <- p_max
n_large_thin1 <- exp_n / p_thin1
lambda_large_thin1 <- poach_prob * n_large_thin1
pp_large_p <- rpoispp(lambda_large_thin1) # inhomogeneous Poisson
pp_thin_p <- rthin(pp_large_p, p_thin1)
pp_tmp <- superimpose(pp_thin_p, pp_large_p)
pp_comp_p <- pp_tmp[-which(multiplicity(pp_tmp) > 1)]
# p(x)-thinning
p_thin2 <- poach_prob * p_max /  max(poach_prob)
n_large_thin2 <- exp_n / mean(p_thin2)
lambda_large_thin2 <- n_large_thin2 / w_area
pp_large_px <- rpoispp(lambda_large_thin2, win = w) # csr
pp_thin_px <- rthin(pp_large_px, p_thin2)
pp_tmp <- superimpose(pp_thin_px, pp_large_px)
pp_comp_px <- pp_tmp[-which(multiplicity(pp_tmp) > 1)]
# P(X)-thinning
dim <- dim(poach_prob)
p_thin3 <- as.im(matrix(runif(dim[1] * dim[2], max = p_max)),
            w, dimyx = c(dim[1], dim[2]))
n_large_thin3 <- exp_n / mean(p_thin3)
lambda_large_thin3 <- poach_prob * n_large_thin3
pp_large_rpx <- rpoispp(lambda_large_thin3) # inhomogeneous Poisson
pp_thin_rpx <- rthin(pp_large_rpx, p_thin3)
pp_tmp <- superimpose(pp_thin_rpx, pp_large_rpx)
pp_comp_rpx <- pp_tmp[-which(multiplicity(pp_tmp) > 1)]

# Quadrat tests for CSR
pp_all <- solist(pp_csr, pp_pois, pp_win,
                  pp_thin_p, pp_thin_px, pp_thin_rpx)
pp_names <- c("CSR", "Poisson", "Sample window",
              "p thinning", "px thinning", "Px thinning")
names(pp_all) <- pp_names
qtest_x2 <- lapply(pp_all, function(x) quadrat.test(x, nx = 4, ny = 3))
qtest_x2_stat <- sapply(qtest_x2, function(x) x$statistic)
qtest_x2_pval <- sapply(qtest_x2, function(x) x$p.value)
qtest_g2 <- lapply(pp_all, function(x) quadrat.test(x, nx = 4, ny = 3, CR = 0))
qtest_g2_stat <- sapply(qtest_g2, function(x) x$statistic)
qtest_g2_pval <- sapply(qtest_g2, function(x) x$p.value)
qtest_t2 <- lapply(pp_all, function(x) quadrat.test(x, 4, 3, CR = -1 / 2))
qtest_t2_stat <- sapply(qtest_t2, function(x) x$statistic)
qtest_t2_pval <- sapply(qtest_t2, function(x) x$p.value)
qtest <- data.frame(X2 = qtest_x2_stat, X2_pval = qtest_x2_pval,
                    G2 = qtest_g2_stat, G2_pval = qtest_g2_pval,
                    T2 = qtest_t2_stat, T2_pval = qtest_t2_pval)
rownames(qtest) <- pp_names
qtest <- round(qtest, 3)

# plot Point processes
pdf(paste0(path, "point_patterns.pdf"), height = 2.5, width = 7)
par(mfrow = c(1, 2), mar = c(1, 1, 1, 0), cex.main = 0.6)
plot(w, main = "Homogeneous Poisson (CSR) Process")
plot(pp_csr, pch = 16, cex = 0.5, col = "steelblue",
  show.window = FALSE, add = TRUE)
mtext(bquote(n == .(npoints(pp_csr)) ~ ", " ~
      bar(lambda) == .(round(intensity(pp_csr), 2))),
      side = 1, line = 0, cex = 0.6)
plot(w, main = "Inhomogeneous Poisson Process")
plot(pp_pois, show.window = FALSE, add = TRUE, pch = 16,
      cex = 0.5, col = "tomato",)
mtext(bquote(n == .(npoints(pp_pois)) ~ ", " ~
      bar(lambda) == .(round(intensity(pp_pois), 2))),
      side = 1, line = 0, cex = 0.6)
dev.off()

# plot window sampling
pdf(paste0(path, "point_patterns_win.pdf"), height = 2, width = 7)
par(mfrow = c(1, 3), mar = c(1, 0, 1, 1) + 0.1, cex.main = 0.8)
plot(w, main = "Inhomogeneous Poisson Process in W")
plot(pp_large_win, show.window = FALSE, add = TRUE, pch = 16, cex = 0.5,
      col = "tomato")
mtext(bquote(n == .(npoints(pp_large_win)) ~ ", " ~
      bar(lambda) == .(round(intensity(pp_large_win), 2))),
      side = 1, line = 0, cex = 0.6)
plot(w, main = "Retain only points in B")
plot(pp_comp_win, show.window = FALSE, add = TRUE, pch = 1, cex = 0.5)
plot(b, add = TRUE)
plot(pp_win, show.window = FALSE, add = TRUE, pch = 16, cex = 0.5,
      col = "tomato")
plot(b, main = "Inhomogeneous Poisson Process in B")
plot(pp_win, show.window = FALSE, add = TRUE, pch = 16, cex = 0.5,
      col = "tomato")
mtext(bquote(n == .(npoints(pp_win)) ~ ", " ~
      bar(lambda) == .(round(intensity(pp_win), 2))),
      side = 1, line = 0, cex = 0.6)
dev.off()

# plot thinned process
# p-thinning of inhomogeneous Poisson
pdf(paste0(path, "point_patterns_thin.pdf"), height = 5.5, width = 7)
par(mfrow = c(3, 3), mar = c(1, 0, 2, 1) + 0.1, cex.main = 0.8)
plot(w, main = "Inhomogeneous Poisson Process")
plot(pp_large_p, show.window = FALSE, add = TRUE, pch = 16,
      cex = 0.5, col = "tomato")
mtext(bquote(n == .(npoints(pp_large_p)) ~ ", " ~
      bar(lambda) == .(round(intensity(pp_large_p), 2))),
      side = 1, line = 0, cex = 0.6)
plot(w, main = expression(paste("Random ", p, "-thinning")))
plot(pp_thin_p, show.window = FALSE, add = TRUE, pch = 16,
      cex = 0.5, col = "tomato")
plot(pp_comp_p, show.window = FALSE, add = TRUE, pch = 1,
      cex = 0.5)
plot(w, main = "Inhomogeneous Poisson Process")
plot(pp_thin_p, show.window = FALSE, add = TRUE, pch = 16,
      cex = 0.5, col = "tomato")
mtext(bquote(n == .(npoints(pp_thin_p)) ~ ", " ~
      bar(lambda) == .(round(intensity(pp_thin_p), 2))),
      side = 1, line = 0, cex = 0.6)
# p(x)-thinning of homogeneous poisson
plot(w, main = "Homogeneous Poisson Process")
plot(pp_large_px, show.window = FALSE, add = TRUE, pch = 16,
      cex = 0.5, col = "steelblue")
mtext(bquote(n == .(npoints(pp_large_px)) ~ ", " ~
      bar(lambda) == .(round(intensity(pp_large_px), 2))),
      side = 1, line = 0, cex = 0.6)
plot(w, main = expression(paste("Random ", p(x), "-thinning")))
plot(pp_thin_px, show.window = FALSE, add = TRUE, pch = 16,
      cex = 0.5, col = "tomato")
plot(pp_comp_px, show.window = FALSE, add = TRUE, pch = 1,
      cex = 0.5)
plot(w, main = "Inhomogeneous Poisson Process")
plot(pp_thin_px, show.window = FALSE, add = TRUE, pch = 16,
      cex = 0.5, col = "tomato")
mtext(bquote(n == .(npoints(pp_thin_px)) ~ ", " ~
      bar(lambda) == .(round(intensity(pp_thin_px), 2))),
      side = 1, line = 0, cex = 0.6)
# P(x)-thinning of inhomogeneous poisson
plot(w, main = "Inhomogeneous Poisson Process")
plot(pp_large_rpx, show.window = FALSE, add = TRUE, pch = 16,
      cex = 0.5, col = "tomato")
mtext(bquote(n == .(npoints(pp_large_rpx)) ~ ", " ~
      bar(lambda) == .(round(intensity(pp_large_rpx), 2))),
      side = 1, line = 0, cex = 0.6)
plot(w, main = expression(paste("Random ", P(x), "-thinning")))
plot(pp_thin_rpx, show.window = FALSE, add = TRUE, pch = 16,
      cex = 0.5, col = "goldenrod")
plot(pp_comp_rpx, show.window = FALSE, add = TRUE, pch = 1,
      cex = 0.5)
plot(w, main = "Cox Process")
plot(pp_thin_rpx, show.window = FALSE, add = TRUE, pch = 16,
      cex = 0.5, col = "goldenrod")
mtext(bquote(n == .(npoints(pp_thin_rpx)) ~ ", " ~
      bar(lambda) == .(round(intensity(pp_thin_rpx), 2))),
      side = 1, line = 0, cex = 0.6)
dev.off()

# plot Kernel densities of point patterns
pdf(paste0(path, "point_patterns_density.pdf"), height = 7, width = 7)
par(mfrow = c(3, 2), mar = c(1, 0, 2, 1) + 0.1, cex.main = 0.8)
for (i in seq_along(pp_all)) {
  plot(density(pp_all[[i]]), main = names(pp_all)[i])
  mtext(bquote(n == .(npoints(pp_all[[i]])) ~ ", " ~
      bar(lambda) == .(round(intensity(pp_all[[i]]), 2))),
      side = 1, line = 0, cex = 0.6)
}
dev.off()

## simulation study - simulate data different ways, vary sample size

iter <- 1000
data_sim <- pp_names
sample_size <- c(50, 100, 200, 300, 400, 500, 1000)
p_win <- 0.8 * b_area / w_area # proportion of pp in sample window
p_max <- 0.8 # max proportion of pp to thin
sim_sample_var <- data.frame()
sim_sample_eval <- data.frame()
for (i in 1:iter) {
  print(paste("iteration: ", i, sep = ""))
  for (j in seq_along(data_sim)) {
    print(paste("data simulation method: ", j, sep = ""))
    for (n in seq_along(sample_size)) {
      if (j == 1) { # csr
        lambda <- sample_size[n] / w_area
        pp <- rpoispp(lambda, win = w)
        ave_lambda <- lambda
        npoints_large  <- npoints(pp)
        inten_large <- intensity(pp)
        sampling <- "Point Process"
        process <- "Homogeneous Poisson"
      } else if (j == 2) { # inhomogeneous poisson pp
        lambda <- poach_prob * sample_size[n]
        pp <- rpoispp(lambda)
        ave_lambda <- mean(lambda)
        npoints_large  <- npoints(pp)
        inten_large <- intensity(pp)
        sampling <- "Point Process"
        process <- "Inhomogeneous Poisson"
      } else if (j == 3) { # sample win of inhomogeneous poisson pp
        n_large <- round(sample_size[n] / p_win)
        lambda <- poach_prob * n_large
        pp_large <- rpoispp(lambda)
        npoints_large  <- npoints(pp_large)
        inten_large <- intensity(pp_large)
        pp <- pp_large[b]
        ave_lambda <- mean(lambda)
        sampling <- "Sample Window"
        process <- "Inhomogeneous Poisson"
      } else if (j == 4) { # p-thin inhomogeneous poisson pp
        p_thin <- p_max
        n_large <- round(sample_size[n] / p_thin)
        lambda <- poach_prob * n_large
        pp_large <- rpoispp(lambda)
        pp <- rthin(pp_large, p_thin)
        ave_lambda <- mean(lambda)
        npoints_large  <- npoints(pp_large)
        inten_large <- intensity(pp_large)
        sampling <- "Thinning"
        process <- "Inhomogeneous Poisson"
      } else if (j == 5) { # p(x)-thin homogeneous poisson pp
        p_thin <- poach_prob * p_max /  max(poach_prob)
        n_large <- round(sample_size[n] / mean(p_thin))
        lambda <- n_large / w_area
        pp_large <- rpoispp(lambda, win = w)
        pp <- rthin(pp_large, p_thin)
        ave_lambda <- lambda
        npoints_large  <- npoints(pp_large)
        inten_large <- intensity(pp_large)
        sampling <- "Thinning"
        process <- "Homogeneous Poisson"
      } else if (j == 6) { # P(x)-thin inhomogeneous poisson pp
        dim <- dim(poach_prob)
        p_thin <- as.im(matrix(runif(dim[1] * dim[2], max = p_max)),
            w, dimyx = c(dim[1], dim[2]))
        n_large <- round(sample_size[n] / mean(p_thin))
        lambda <- poach_prob * n_large
        pp_large <- rpoispp(lambda)
        pp <- rthin(pp_large, p_thin)
        ave_lambda <- mean(lambda)
        npoints_large  <- npoints(pp_large)
        inten_large <- intensity(pp_large)
        sampling <- "Thinning"
        process <- "Inhomogeneous Poisson"
      }
      fit <- ppm(pp ~ ., covariates = covariates)
      fit_csr <- ppm(pp ~ 1)
      fit_qt <- quadrat.test(fit, nx = 4, ny = 3)
      fit_csr_qt <- quadrat.test(fit_csr, nx = 4, ny = 3)
      variables <- data.frame(iter = i,
                              data_sim = data_sim[j],
                              sample_size = sample_size[n],
                              variable = c("Intercept", descrip),
                              coef = summary(fit)$coef[, 1],
                              se = summary(fit)$coef[, 2],
                              wald = summary(fit)$coef[, 6])
      sim_sample_var <- rbind(sim_sample_var, variables)
      eval <- data.frame(iter = i,
                          data_sim = data_sim[j],
                          Sampling = sampling,
                          Process = process,
                          sample_size = sample_size[n],
                          npoints = npoints(pp),
                          n_large = npoints_large,
                          p_win = p_win,
                          p_max = p_max,
                          ave_lambda = ave_lambda,
                          ave_intensity = intensity(pp),
                          inten_large = inten_large,
                          mae = sum(abs(residuals(fit)$val)) / npoints(pp),
                      mae_csr = sum(abs(residuals(fit_csr)$val)) / npoints(pp),
                      gof_pval = fit_qt$p.value,
                      gof_pval_csr = fit_csr_qt$p.value)
      sim_sample_eval <- rbind(sim_sample_eval, eval)
    }
  }
}
# factor variables
sim_sample_eval$sample_size <- as.factor(sim_sample_eval$sample_size)
sim_sample_eval$data_sim <- factor(sim_sample_eval$data_sim, levels = data_sim)
sim_sample_eval$Sampling <- factor(sim_sample_eval$Sampling,
                    levels = c("Point Process", "Sample Window", "Thinning"))
sim_sample_eval$Process <- factor(sim_sample_eval$Process,
                            levels = c("Homogeneous Poisson",
                            "Inhomogeneous Poisson"))

sim_labs <- c("Homogeneous Poisson (CSR)", "Inhomogeneous Poisson",
              "Window sample of Inhomogeneous Poisson",
              "p-Thinning of Inhomogeneous Poisson",
              "p(x)-Thinning of Homogeneous Poisson",
              "P(x)-Thinning of Inhomogeneous Poisson")
names(sim_labs) <- data_sim

# plot gof rejection rate
sim_sample_eval["gof_cov"] <- ifelse(sim_sample_eval$gof_pval < 0.05,
                                      "Reject", "Do not reject")
sim_sample_eval["gof_csr"] <- ifelse(sim_sample_eval$gof_pval_csr < 0.05,
                                      "Reject", "Do not reject")
sim_sample_eval$gof_cov <- factor(sim_sample_eval$gof_cov,
                                  levels = c("Reject", "Do not reject"))
sim_sample_eval$gof_csr <- factor(sim_sample_eval$gof_csr,
                                  levels = c("Reject", "Do not reject"))

# bar plots for gof
s_gof_csr <- ggplot(sim_sample_eval, aes(x = sample_size,
                      fill = gof_csr)) +
              geom_bar(position =   position_dodge()) +
              facet_wrap(~data_sim, ncol = 3) +
              labs(x = "Sample size", y = "Count") +
              theme(legend.position = "top",
              axis.title.x = element_blank()) +
              scale_fill_discrete(name = "H0: CSR model",
                guide = guide_legend(nrow = 1))
s_gof_cov <- ggplot(sim_sample_eval, aes(x = sample_size,
                      fill = gof_cov)) +
              geom_bar(position =   position_dodge()) +
              facet_wrap(~data_sim, ncol = 3) +
              labs(x = "Sample size", y = "Count") +
              theme(legend.position = "top") +
              scale_fill_discrete(name = "H0: Covariate model",
                guide = guide_legend(nrow = 1))
s_gof <-  s_gof_csr / s_gof_cov
ggsave(filename = "sim_sample_gof.pdf", plot = s_gof,
        device = "pdf", path = path, width = 7, height = 9,
        units = "in", dpi = 500, limitsize = TRUE)


# plots of mean absolute error of CSR fit
s_csr_mae_box <- ggplot(sim_sample_eval, aes(y = sample_size, x = mae_csr)) +
              geom_boxplot() +
              facet_wrap(~data_sim, ncol = 2,
                labeller = labeller(data_sim = sim_labs)) +
              labs(y = "Sample size") +
              theme(axis.title.x = element_blank())
s_csr_mae_dens <- ggplot(sim_sample_eval, aes(mae_csr, colour = sample_size)) +
                geom_density() +
              facet_wrap(~data_sim, ncol = 2,
                labeller = labeller(data_sim = sim_labs)) +
                labs(x = "Mean absolute error", y = "Density") +
                theme(legend.position = c(0.795, 0.94),
                  legend.title = element_text(size = 7),
                  legend.text = element_text(size = 7),
                  legend.key.size = unit(0.2, "cm"),
                  legend.background = element_blank(),
                  legend.box.background = element_rect(colour = "black")) +
                scale_colour_discrete(name = "Sample size",
                  guide = guide_legend(nrow = 1))
s_csr_mae <- s_csr_mae_box / s_csr_mae_dens
ggsave(filename = "sim_sample_mae_csr.pdf", plot = s_csr_mae,
        device = "pdf", path = path, width = 7, height = 9,
        units = "in", dpi = 500, limitsize = TRUE)

# plots of mean absolute error
s_mae_box <- ggplot(sim_sample_eval, aes(y = sample_size, x = mae)) +
              geom_boxplot() +
              facet_wrap(~data_sim, ncol = 2,
                labeller = labeller(data_sim = sim_labs)) +
              labs(y = "Sample size") +
              theme(axis.title.x = element_blank())
s_mae_dens <- ggplot(sim_sample_eval, aes(mae, colour = sample_size)) +
                geom_density() +
                facet_wrap(~data_sim, ncol = 2,
                labeller = labeller(data_sim = sim_labs)) +
                labs(x = "Mean absolute error", y = "Density") +
                theme(legend.position = c(0.795, 0.94),
                  legend.title = element_text(size = 7),
                  legend.text = element_text(size = 7),
                  legend.key.size = unit(0.2, "cm"),
                  legend.background = element_blank(),
                  legend.box.background = element_rect(colour = "black")) +
                scale_colour_discrete(name = "Sample size",
                  guide = guide_legend(nrow = 1))
s_mae <- s_mae_box / s_mae_dens
ggsave(filename = "sim_sample_mae.pdf", plot = s_mae,
        device = "pdf", path = path, width = 7, height = 9,
        units = "in", dpi = 500, limitsize = TRUE)

# boxplots for n = best_n
n <- 200
ss <- sim_sample_eval[sim_sample_eval$sample_size == n, ]
s_mae_box_n <- ggplot(ss, aes(x = data_sim, y = mae)) +
              geom_boxplot() +
              labs(x = "Data simulation", y = "") +
              coord_flip()
s_mae_dens_n <- ggplot(ss, aes(mae, colour = data_sim)) +
                geom_density() +
                labs(x = "Mean absolute error", y = "Density") +
                theme(legend.position = c(0.8, 0.7),
                  legend.title = element_text(size = 7),
                  legend.text = element_text(size = 7),
                  legend.key.size = unit(0.2, "cm"),
                  legend.background = element_blank(),
                  legend.box.background = element_rect(colour = "black")) +
                scale_colour_discrete(name = "Data simulation",
                  guide = guide_legend(ncol = 1))
s_mae_n <- s_mae_box_n / s_mae_dens_n
ggsave(filename = "sim_sample_mae_n.pdf", plot = s_mae_n,
        device = "pdf", path = path, width = 5, height = 5,
        units = "in", dpi = 500, limitsize = TRUE)

# plot npoints distribution, choose best sample size
n <- 200
sim_sample_ss <- sim_sample_eval[sim_sample_eval$sample_size == n, ]

pdf(file = paste0(path, "sim_sample_distr.pdf"), width = 7, height = 5)
par(mfrow = c(3, 2), oma = c(3, 3, 0, 0) + 0.1, mar = c(1, 1, 1, 1.5) + 0.2,
    mgp = c(3, 0.2, 0))
  x_tck <- seq(140, 260, 20)
  y_tck <- seq(0, 0.03, 0.01)
  for (i in seq_along(data_sim)) {
    x <- sim_sample_ss[sim_sample_ss$data_sim == data_sim[i], "npoints"]
    x_labs <- if (i %in% 1:4) FALSE else x_tck
    y_labs <- if (i %in% c(2, 4, 6)) FALSE else y_tck
    plot(density(x), lwd = 2, xlab = "", ylab = "", xaxt = "n", yaxt = "n",
      main = sim_labs[i], cex.main = 0.9, xlim = c(140, 260), ylim = c(0, 0.03))
    axis(1, at = x_tck, labels = x_labs, cex.axis = 0.8, tck = -0.02)
    axis(2, at = y_tck, labels = y_labs, cex.axis = 0.8, tck = -0.02)
    abline(h = y_tck, col = "grey", lty = 1)
    abline(v = x_tck, col = "grey", lty = 1)
    points(x, dpois(x, mean(x)), col = rgb(1, 0, 0, 0.2), pch = 1, cex = 0.5)
}
title(xlab = "Number of Points", ylab = "Density", outer = TRUE, line = 1)
dev.off()

# boxplots of Z scores with 95% normal CI
# choose best data_sim
best_data <- c("Sample window", "p thinning")
sim_sample_var$sample_size <- as.factor(sim_sample_var$sample_size)
sim_sample_var$variable <- factor(sim_sample_var$variable,
                                  levels = c("Intercept", descrip))
sim_sample_var$data_sim <- factor(sim_sample_var$data_sim,
                                  levels = data_sim)
sim_sample_var_ss <- sim_sample_var[sim_sample_var$data_sim %in% best_data, ]
labs <- c("intercept", names)
names(labs) <- c("Intercept", descrip)

s_var_plot <- ggplot(sim_sample_var_ss, aes(x = sample_size, y = wald)) +
                geom_boxplot() +
                annotate("rect", xmin = 0, xmax = 8, ymin = -2,
                         ymax = 2, alpha = 0.2, fill = "blue") +
                facet_grid(vars(variable), vars(data_sim),
                labeller = labeller(variable = labs)) +
                labs(x = "Sample size",
                  y = "Wald statistic to test for zero coefficient") +
                theme(plot.title = element_text(hjust = 0.5))
ggsave(filename = "sim_sample_variables.pdf", plot = s_var_plot,
              device = "pdf", path = path, width = 6, height = 9,
              units = "in", dpi = 500, limitsize = TRUE)

## simulation study - varying models

# choose best data simulation and sample size
iter <- 1000
n <- 200
p_thin <- 0.8
p_win <- 0.8 * b_area / w_area
best_data <- c("Sample window", "p thinning")
models <- c("CSR", "LIN", "QD", "COV", "L+C", "Q+C")
sim_model_eval <- data.frame()
set.seed(12345)
for (i in 1:iter) {
  print(paste("iteration: ", i, sep = ""))
  for (j in seq_along(data_sim)) {
    if (j == 1) {
      n_large <- round(n / p_win)
      lambda <- poach_prob * n_large
      pp <- rpoispp(lambda)
      sample <- pp_large[b]
    } else if (j == 2) {
        n_large <- round(n / p_thin)
        lambda <- poach_prob * n_large
        pp <- rpoispp(lambda)
        sample <- rthin(pp_large, p_thin)
    }
    for (m in seq_along(models)) {
      if (m == 1) {
        fit <- ppm(sample ~ 1)
      } else if (m == 2) {
        fit <- ppm(sample ~ x + y)
      } else if (m == 3) {
        fit <- ppm(sample ~ polynom(x, y, 2))
      } else if (m == 4) {
        fit <- ppm(sample ~ ., covariates = covariates)
      } else if (m == 5) {
        fit <- ppm(sample ~ x + y + ., covariates = covariates)
      } else if (m == 6) {
        fit <- ppm(sample ~ polynom(x, y, 2) + ., covariates = covariates)
      }
      eval <- data.frame(iter = i,
                        data_sim = data_sim[j],
                        model = models[m],
                        npoints = npoints(sample),
                        n_large = npoints(pp),
                        p_thin = p_thin,
                        p_win = p_win,
                        ave_lambda = mean(lambda),
                        ave_intensity = intensity(sample),
                        ave_int_large = intensity(pp),
                        mae = sum(abs(residuals(fit)$val)) / npoints(sample),
                        mse = sum((residuals(fit)$val)^2) / npoints(sample),
                        loglik = logLik(fit)[1],
                        deviance = deviance(fit),
                        aic = AIC(fit),
                        singular = ifelse(length(vcov(fit)) > 0, 0, 1))
      sim_model_eval <- rbind(sim_model_eval, eval)
    }
  }
}


# plot to check singular cov matrix
sim_model_eval$model <- factor(sim_model_eval$model, levels = models)
bar <- ggplot(sim_model_eval, aes(x = model, y = singular)) +
        geom_bar(stat = "identity")
ggsave(filename = "sim_model_singular.pdf", plot = bar,
        device = "pdf", path = path, width = 6, height = 4,
        units = "in", dpi = 500, limitsize = TRUE)

# plots of mean absolute error
sim_model_eval$singular <- factor(sim_model_eval$singular)
ss <- sim_model_eval[sim_model_eval$data_sim %in% best_data, ]
m_mae_box <- ggplot(ss, aes(x = model, y = mae, color = singular)) +
              geom_boxplot() +
              coord_flip() +
              labs(x = "Model", y = "") +
              theme_bw() +
              facet_wrap(~ data_sim, ncol = 2) +
                theme(legend.position = c(0.145, 0.12),
                  legend.title = element_text(size = 8),
                  legend.text = element_text(size = 8),
                  legend.key.size = unit(0.25, "cm"),
                  legend.background = element_blank(),
                  legend.box.background = element_rect(colour = "black")) +
                scale_colour_discrete(name = "Covariance matrix",
                  labels = c("Nonsingular", "Singular"),
                  guide = guide_legend(nrow = 1))
m_mae_dens <- ggplot(ss, aes(mae, colour = model)) +
                geom_density() +
                labs(x = "Mean absolute error", y = "Density") +
                theme_bw() +
              facet_wrap(~ data_sim, ncol = 2) +
                theme(legend.position = c(0.23, 0.87),
                  legend.title = element_text(size = 8),
                  legend.text = element_text(size = 8),
                  legend.key.size = unit(0.25, "cm"),
                  legend.background = element_blank(),
                  legend.box.background = element_rect(colour = "black")) +
                scale_colour_discrete(name = "Model",
                  guide = guide_legend(nrow = 1))
m_mae <- m_mae_box / m_mae_dens 
ggsave(filename = "sim_model_mae.pdf", plot = m_mae,
        device = "pdf", path = path, width = 7, height = 7,
        units = "in", dpi = 500, limitsize = TRUE)

# plots of AIC
m_aic_box <- ggplot(ss, aes(x = model, y = aic, color = singular)) +
              geom_boxplot() +
              coord_flip() +
              facet_wrap(~ data_sim, ncol = 2) +
              labs(x = "Model", y = "") +
              theme_bw() +
                theme(legend.position = c(0.145, 0.12),
                  legend.title = element_text(size = 8),
                  legend.text = element_text(size = 8),
                  legend.key.size = unit(0.25, "cm"),
                  legend.background = element_blank(),
                  legend.box.background = element_rect(colour = "black")) +
                scale_colour_discrete(name = "Covariance matrix",
                  labels = c("Nonsingular", "Singular"),
                  guide = guide_legend(nrow = 1))
m_aic_dens <- ggplot(ss, aes(aic, colour = model)) +
                geom_density() +
              facet_wrap(~ data_sim, ncol = 2) +
                labs(x = "AIC", y = "Density") +
                theme_bw() +
                theme(legend.position = c(0.23, 0.87),
                  legend.title = element_text(size = 8),
                  legend.text = element_text(size = 8),
                  legend.key.size = unit(0.25, "cm"),
                  legend.background = element_blank(),
                  legend.box.background = element_rect(colour = "black")) +
                scale_colour_discrete(name = "Model",
                  guide = guide_legend(nrow = 1))
m_aic <- m_aic_box / m_aic_dens +
          plot_layout(widths = c(1, 1), heights = c(1, 1))
ggsave(filename = "sim_model_aic.pdf", plot = m_aic,
        device = "pdf", path = path, width = 7, height = 7,
        units = "in", dpi = 500, limitsize = TRUE)

## final models, sample size 400

set.seed(654321)
n <- 400
p_thin <- 0.8
n_large <- round(n / p_thin)
pp <- rpoint(n = n_large, f = poach_image)
sample <- rthin(pp, p_thin)
pp <- sample

# plots and table of quadrats for covariates (first plot point pattern)
cov_quad <- data.frame()
fit_uni <- ppm(pp ~ 1)
pdf(paste(path, "cov_quadrats.pdf", sep = ""), height = 9, width = 7)
par(mfrow = c(4, 2), mar = c(1, 0, 1, 1) + 0.1, cex.main = 0.8)
plot(poach_prob * n,
  main = paste("Sample of pp generated from poacher probability with n=",
    n, sep = ""),
  ribsep = 0.05)
plot(pp, add = TRUE)
for (i in seq_along(covariates)) {
  z <- covariates[[i]]
  v <- quantess(pp, covariates[[i]], 4)
  qb <- quadratcount(pp, tess = v)
  cov <- substring(descrip[i], 1, nchar(descrip[i]) - 5)
  plot(qb, main = paste("Quadrat counts for quantiles of ", cov, sep = ""),
    ribsep = 0.05)
  cov_ppm <- ppm(pp ~ ., covariates = covariates[i])
  cov_entry <- data.frame(Q1 = qb[1], Q2 = qb[2], Q3 = qb[3], Q4 = qb[4],
    Two_sided = quadrat.test(qb, alternative = "two.sided")$p.value,
    Regular = quadrat.test(qb, alternative = "regular")$p.value,
    Clustered = quadrat.test(qb, alternative = "clustered")$p.value,
    Berman_Z1 = berman.test(pp, z)$p.value,
    Berman_Z2 = berman.test(pp, z, "Z2")$p.value,
    LRT = anova(fit_uni, cov_ppm, test = "LRT")$"Pr(>Chi)"[2],
    row.names = cov)
  cov_quad <- rbind(cov_quad, cov_entry)
}
dev.off()

round(cov_quad, 5)

# intensity estimate rho(cov)
pdf(paste(path, "cov_rhohat.pdf", sep = ""), height = 9, width = 7)
par(mfrow = c(4, 2), mar = c(2, 2, 2, 1) + 0.1, cex.main = 0.8)
for (i in seq_along(covariates)) {
  cov <- substring(descrip[i], 1, nchar(descrip[i]) - 5)
  plot(rhohat(pp, covariates[[i]]),
    main = paste("Intensity estimate for ", cov, sep = ""))
}
dev.off()

# fit different models
fit_csr <- ppm(pp ~ 1)
fit_cov <- ppm(pp ~ ., covariates = covariates)
fit_lin <- ppm(pp ~ x + y)
fit_qd <- ppm(pp ~ polynom(x, y, 2))
fit_lnc <- ppm(pp ~ x + y + ., covariates = covariates)
fit_qnc <- ppm(pp ~ polynom(x, y, 2) + ., covariates = covariates)
step_lnc <- step(fit_lnc)
step_cov <- step(fit_cov)

fit_models <- list(fit_csr, fit_cov, fit_lnc, fit_qnc, step_cov, step_lnc)

# print fitted models
for (m in fit_models) {
  print(m)
  cat("\n\n\n\n\n")
}

for (m in fit_models) {
  print(m$trend)
  cat("\n")
  cat(paste("AIC: ", AIC(m), sep = ""))
  cat("\n")
  cat(paste("MAE: ", sum(abs(residuals(m)$val)) / npoints(pp)), sep = "")
  cat("\n")
  cat(paste("GOF p-value: ", quadrat.test(m, nx = 4, ny = 4)$p.value, sep = ""))
  cat("\n\n")
}

# plot fitted intensity
pdf(paste(path, "fit_intensity.pdf", sep = ""), height = 7, width = 7)
par(mfrow = c(3, 2), mar = c(1, 0, 1, 1) + 0.1, cex.main = 0.8, cex.axis = 0.8)
plot(poach_prob * n, main = "Poacher intensity", ribsep = 0.05)
plot(fit_cov, cif = FALSE, se = FALSE, superimpose = FALSE,
  main = "Fitted intensity, covariate model", ribsep = 0.05)
plot(fit_lnc, cif = FALSE, se = FALSE, superimpose = FALSE,
  main = "Fitted intensity, linear + covariate model", ribsep = 0.05)
plot(fit_qnc, cif = FALSE, se = FALSE, superimpose = FALSE,
  main = "Fitted intensity, quadratic + covariate model", ribsep = 0.05)
plot(step_cov, cif = FALSE, se = FALSE, superimpose = FALSE,
  main = "Fitted intensity, stepwise deletion from covariate model",
  ribsep = 0.05)
plot(step_lnc, cif = FALSE, se = FALSE, superimpose = FALSE,
  main = "Fitted intensity, stepwise deletion from linear + covariate model",
  ribsep = 0.05)
dev.off()

# plot residual contours
pdf(paste(path, "fit_residuals.pdf", sep = ""), height = 7, width = 7)
par(mfrow = c(3, 2), mar = c(1, 0, 1, 1) + 0.1, cex.main = 0.8, cex.axis = 0.8)
diagnose.ppm(fit_csr, which = "smooth", type = "raw",
  main = "Smoothed residuals, CSR model")
diagnose.ppm(fit_cov, which = "smooth", type = "raw",
  main = "Smoothed residuals, covariate model")
diagnose.ppm(fit_lnc, which = "smooth", type = "raw",
  main = "Smoothed residuals, linear + covariate model")
diagnose.ppm(fit_qnc, which = "smooth", type = "raw",
  main = "Smoothed residuals, quadratic + covariate model")
diagnose.ppm(step_cov, which = "smooth", type = "raw",
  main = "Smoothed residuals, stepwise deletion from covariate model")
diagnose.ppm(step_lnc, which = "smooth", type = "raw",
  main = "Smoothed residuals, stepwise deletion from linear + covariate model")
dev.off()

# save best fit as polygon grid shapefile
best_fit <- fit_lnc
image_fit <- predict(best_fit, locations = poach_image,
  covariates = images, type = "intensity") # fitted intensity image
prob_fit <- image_fit / npoints(pp) # probability image
raster_fit <- raster(prob_fit) # probability raster
crs(raster_fit) <- crs(grid_proj) # set raster crs
vector_fit <- as(raster_fit, "SpatialPolygonsDataFrame") # convert to polygon
vector_fit <- spTransform(vector_fit, crs(grid_gps)) # reproject to gps
vector_fit <- st_as_sf(vector_fit) # convert to sf
st_write(vector_fit, paste(path, "shapefiles/ppm_fit.shp", sep = ""),
    driver = "ESRI Shapefile")  # save shapefile

