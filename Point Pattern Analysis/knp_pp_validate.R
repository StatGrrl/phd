library(sf)
library(spatstat)
library(raster)
path <- "C:/Users/lisak/OneDrive/PhD2.0/Coding/Simulations/spatstat/final_ex1/"
load(paste0(path, "knp_pp_model.RData"))

rm(fit_all_df, z, alter, cov_dir, fit, fit_all, fit_all_names, fit_back_cov,
    fit_back_cov_inter, fit_back_cov_poly_inter, fit_back_cov_poly,
    fit_border, fit_camps, fit_cov_inter, fit_cov_poly_inter, fit_cov_poly,
    fit_dams, fit_forw_cov, fit_forw_cov_inter, fit_forw_cov_poly_inter,
    fit_forw_cov_poly, fit_gates, fit_lrt, fit_picnic, fit_polys, fit_roads,
    fit_water, fit_z1, fit_z2, gof, i, panel, pval, qt, x, xlab, ylab)

models <- list(fit_cov, fit_csr)
models <- c(best, models)
rm(best, fit_cov, fit_csr)

# relative intensity kernel smoothing
xlab <- "Westing distance (km)"
ylab <- "Southing distance (km)"
panel <- c("(a)", "(b)", "(c)", "(d)", "(e)", "(f)")
pdf(paste0(path, "eval_relative_intensity.pdf"), width = 7, height = 7)
par(mfrow = c(3, 2), mar = c(2, 2, 2, 3), oma = c(2, 2, 0, 1))
for (i in seq_along(models)) {
    fitted <- fitted(models[[i]], dataonly = TRUE)
    plot(reflect(density(sample, weights = 1 / fitted)), main = "",
        axes = TRUE, xlab = "", ylab = "", cex.axis = 0.8, ribsep = 0.05,
        ribside = "right")
    mtext("Relative Intensity", side = 4, line = 1.5, cex = 0.8)
    mtext(panel[i], side = 3, line = 0, cex = 0.8)
}
mtext(xlab, side = 1, line = 0, outer = TRUE)
mtext(ylab, side = 2, line = 0, outer = TRUE)
dev.off()

# plot residual contours
pdf(paste(path, "eval_residuals_smooth.pdf", sep = ""), height = 7, width = 7)
par(mfrow = c(3, 2), mar = c(1, 0, 1, 1), cex.main = 0.8)
for (i in seq_along(models)) {
    diagnose.ppm(models[[i]], which = "smooth", type = "raw",
        main = "")
    #mtext(panel[i], side = 3, line = 0, cex = 0.8)
}
dev.off()

# lurking variable plots
model_cov <- list(c("roads", "camps", "picnic", "gates", "border"),
                c("camps", "border", "water"),
                c("roads", "gates", "water"),
                c("camps", "border", "water"),
                 names(covariates))
xlab <- "Distance (km)"
ylab <- "Cumulative raw residuals"
j <- 1
pdf(paste(path, "eval_res_lurking_a.pdf", sep = ""), height = 4, width = 7)
par(mfrow = c(2, 3), mar = c(2, 2, 1, 1), oma = c(2, 2, 0, 0))
for (i in seq_along(model_cov[[j]])) {
    z <- covariates[[model_cov[[j]][i]]]
    lurking(models[[j]], z, type = "raw", main = "", xlab = "", ylab = "",
        cex.axis = 0.8)
}
mtext(xlab, side = 1, line = 0, outer = TRUE, cex = 0.8)
mtext(ylab, side = 2.5, line = 0, outer = TRUE, cex = 0.8)
dev.off()
j <- 2
pdf(paste(path, "eval_res_lurking_b.pdf", sep = ""), height = 2, width = 7)
par(mfrow = c(1, 3), mar = c(2, 2, 1, 1), oma = c(2, 2, 0, 0))
for (i in seq_along(model_cov[[j]])) {
    z <- covariates[[model_cov[[j]][i]]]
    lurking(models[[j]], z, type = "raw", main = "", xlab = "", ylab = "",
        cex.axis = 0.8)
}
mtext(xlab, side = 1, line = 0, outer = TRUE, cex = 0.8)
mtext(ylab, side = 2.5, line = 0, outer = TRUE, cex = 0.8)
dev.off()
j <- 3
pdf(paste(path, "eval_res_lurking_c.pdf", sep = ""), height = 2, width = 7)
par(mfrow = c(1, 3), mar = c(2, 2, 1, 1), oma = c(2, 2, 0, 0))
for (i in seq_along(model_cov[[j]])) {
    z <- covariates[[model_cov[[j]][i]]]
    lurking(models[[j]], z, type = "raw", main = "", xlab = "", ylab = "",
        cex.axis = 0.8)
}
mtext(xlab, side = 1, line = 0, outer = TRUE, cex = 0.8)
mtext(ylab, side = 2.5, line = 0, outer = TRUE, cex = 0.8)
dev.off()
j <- 4
pdf(paste(path, "eval_res_lurking_d.pdf", sep = ""), height = 2, width = 7)
par(mfrow = c(1, 3), mar = c(2, 2, 1, 1), oma = c(2, 2, 0, 0))
for (i in seq_along(model_cov[[j]])) {
    z <- covariates[[model_cov[[j]][i]]]
    lurking(models[[j]], z, type = "raw", main = "", xlab = "", ylab = "",
        cex.axis = 0.8)
}
mtext(xlab, side = 1, line = 0, outer = TRUE, cex = 0.8)
mtext(ylab, side = 2.5, line = 0, outer = TRUE, cex = 0.8)
dev.off()
j <- 5
panel <- c("(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)")
pdf(paste(path, "eval_res_lurking_e.pdf", sep = ""), height = 9, width = 7)
par(mfrow = c(4, 2), mar = c(2, 2, 2, 1), oma = c(2, 2, 0, 0), cex.main = 0.9)
for (i in seq_along(model_cov[[j]])) {
    z <- covariates[[model_cov[[j]][i]]]
    lurking(models[[j]], z, type = "raw", main = "", xlab = "", ylab = "",
        cex.axis = 0.8)
}
mtext(xlab, side = 1, line = 0, outer = TRUE, cex = 0.8)
mtext(ylab, side = 2.5, line = 0, outer = TRUE, cex = 0.8)
dev.off()


# partial residual plots
xlab <- "Distance (km)"
ylab <- "Log Intensity"
j <- 1
pdf(paste(path, "eval_res_partial_a.pdf", sep = ""), height = 4, width = 7)
par(mfrow = c(2, 3), mar = c(2, 2, 1, 1), oma = c(2, 2, 0, 0))
for (i in seq_along(model_cov[[j]])) {
    plot(parres(models[[j]], model_cov[[j]][i]), main = "",
        xlab = "", ylab = "", cex.axis = 0.8, legend=FALSE)
}
mtext(xlab, side = 1, line = 0, outer = TRUE, cex = 0.8)
mtext(ylab, side = 2.5, line = 0, outer = TRUE, cex = 0.8)
dev.off()
j <- 2
pdf(paste(path, "eval_res_partial_b.pdf", sep = ""), height = 2, width = 7)
par(mfrow = c(1, 3), mar = c(2, 2, 1, 1), oma = c(2, 2, 0, 0))
for (i in seq_along(model_cov[[j]])) {
    plot(parres(models[[j]], model_cov[[j]][i]), main = "",
        xlab = "", ylab = "", cex.axis = 0.8, legend=FALSE)
}
mtext(xlab, side = 1, line = 0, outer = TRUE, cex = 0.8)
mtext(ylab, side = 2.5, line = 0, outer = TRUE, cex = 0.8)
dev.off()
j <- 3
pdf(paste(path, "eval_res_partial_c.pdf", sep = ""), height = 2, width = 7)
par(mfrow = c(1, 3), mar = c(2, 2, 1, 1), oma = c(2, 2, 0, 0))
for (i in seq_along(model_cov[[j]])) {
    plot(parres(models[[j]], model_cov[[j]][i]), main = "",
        xlab = "", ylab = "", cex.axis = 0.8, legend=FALSE)
}
mtext(xlab, side = 1, line = 0, outer = TRUE, cex = 0.8)
mtext(ylab, side = 2.5, line = 0, outer = TRUE, cex = 0.8)
dev.off()
j <- 4
pdf(paste(path, "eval_res_partial_d.pdf", sep = ""), height = 4, width = 6)
par(mfrow = c(1, 1), mar = c(2, 2, 1, 1), oma = c(2, 2, 0, 0))
plot(parres(models[[j]], "border"), main = "",
        xlab = "", ylab = "", cex.axis = 0.8, legend=FALSE)
mtext(xlab, side = 1, line = 0, outer = TRUE, cex = 0.8)
mtext(ylab, side = 2.5, line = 0, outer = TRUE, cex = 0.8)
dev.off()
j <- 5
pdf(paste(path, "eval_res_partial_e.pdf", sep = ""), height = 9, width = 7)
par(mfrow = c(4, 2), mar = c(2, 2, 2, 1), oma = c(2, 2, 0, 0), cex.main = 0.9)
for (i in seq_along(model_cov[[j]])) {
    plot(parres(models[[j]], model_cov[[j]][i]), main = "",
        xlab = "", ylab = "", cex.axis = 0.8, legend=FALSE)
}
mtext(xlab, side = 1, line = 0, outer = TRUE, cex = 0.8)
mtext(ylab, side = 2.5, line = 0, outer = TRUE, cex = 0.8)
dev.off()

# save best fit as polygon grid shapefile
best_fit <- models[[1]]
image_fit <- predict(best_fit, locations = poach_prob,
  covariates = covariates, type = "intensity") # fitted intensity image
image_adj <- image_fit / p # adjusted intensity for thinning
prob_fit <- image_fit / npoints(pp) # probability image
raster_fit <- raster(prob_fit) # probability raster
crs(raster_fit) <- crs(grid_proj) # set raster crs
vector_fit <- as(raster_fit, "SpatialPolygonsDataFrame") # convert to polygon
vector_fit <- spTransform(vector_fit, crs(grid_gps)) # reproject to gps
vector_fit <- st_as_sf(vector_fit) # convert to sf
st_write(vector_fit, paste(path, "shapefiles/ppm_fit.shp", sep = ""),
    driver = "ESRI Shapefile")  # save shapefile

save.image(file = paste0(path, "knp_pp_eval.RData"))
