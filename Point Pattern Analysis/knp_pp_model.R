library(sf)
library(spatstat)
library(ggplot2)
theme_set(theme_bw())
library(patchwork)

path <- "C:/Users/lisak/OneDrive/PhD2.0/Coding/Simulations/spatstat/final_ex1/"
load(paste0(path, "knp_pp_explore.RData"))
rm(cov_entry, cov_quad, kden, pred, qb, qc, rh, tmp, z, cov_ppm, fit_uni, i,
  intensity_plots, n_large, panel, qt, rlab, v, xlab, ylab)

fit_csr <- ppm(sample ~ 1, covariates = covariates)
fit_cov <- ppm(sample ~ ., covariates = covariates)
alter <- c(rep("greater", 4), rep("less", 3))
cov_df <- data.frame()
for (i in seq_along(covariates)) {
  z <- covariates[[i]]
  fit <- ppm(sample ~ ., covariates = covariates[i])
  cov_entry <- data.frame(p = length(coef(fit)),
    intercept = summary(fit)$coef[1, 1],
    coef = summary(fit)$coef[2, 1],
    se = summary(fit)$coef[2, 2],
    pvalue = pnorm(abs(summary(fit)$coef[2, 6]), lower = FALSE) * 2,
    pvalue_low = pnorm(summary(fit)$coef[2, 6], lower = (!cov_dir)[i]),
    LRT = anova(fit_csr, fit, test = "LRT")$"Pr(>Chi)"[2],
    Berman_Z1 = berman.test(fit_csr, z)$p.value,
    Berman_Z2 = berman.test(fit_csr, z, "Z2")$p.value,
    z1_alt = berman.test(fit_csr, z, alternative = alter[i])$p.value,
    z2_alt = berman.test(fit_csr, z, "Z2", alternative = alter[i])$p.value,
    Chi2_df = quadrat.test(fit, nx = 4, ny = 3)$parameter,
    Chi2_pval = quadrat.test(fit, nx = 4, ny = 3)$p.value,
    AIC = AIC(fit),
    MAE = sum(abs(residuals(fit)$val)) / n,
    full_coef = summary(fit_cov)$coef[i + 1, 1],
    full_pvalue = pnorm(abs(summary(fit_cov)$coef[i + 1, 6]),
                        lower = FALSE) * 2,
    full_pvalue_low = pnorm(summary(fit_cov)$coef[i + 1, 6],
                            lower = (!cov_dir)[i]),
    row.names = names(covariates)[i])
  cov_df <- rbind(cov_df, cov_entry)
}
fit_csr
fit_cov
round(vcov(fit_cov, what = "corr"), 2)
round(cov_df, 3)

fit_z1 <- ppm(sample ~ roads + camps + water, covariates = covariates)
fit_z2 <- ppm(sample ~ roads + camps + border + dams + water,
            covariates = covariates)
fit_lrt <- update(fit_cov, ~ . - gates)


fit_roads <- ppm(sample ~ polynom(roads, 3), covariates = covariates)
fit_camps <- ppm(sample ~ polynom(camps, 3), covariates = covariates)
fit_picnic <- ppm(sample ~ polynom(picnic, 3), covariates = covariates)
fit_gates <- ppm(sample ~ polynom(gates, 3), covariates = covariates)
fit_border <- ppm(sample ~ polynom(border, 3), covariates = covariates)
fit_dams <- ppm(sample ~ polynom(dams, 3), covariates = covariates)
fit_water <- ppm(sample ~ polynom(water, 3), covariates = covariates)
fit_polys <- anylist(fit_roads, fit_camps, fit_picnic, fit_gates, fit_border,
                     fit_dams, fit_water)
xlab <- "Distance (km)"
ylab <- "Intensity Estimate"
panel <- c("(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)")
pdf(paste0(path, "model_polys_effect.pdf"), width = 7, height = 9)
par(mfrow = c(4, 2), mar = c(2, 2, 1, 1), cex.main = 1, oma = c(3, 3, 0, 0))
for (i in seq_along(fit_polys)) {
  plot(effectfun(fit_polys[[i]], names(covariates)[i], se.fit = TRUE),
      main = "", legend = FALSE)
  abline(h = 0.09, col = "red", lty = 2)
}
mtext(xlab, side = 1, line = 0.5, cex = 0.8, outer = TRUE)
mtext(ylab, side = 2, line = 1, cex = 0.8, outer = TRUE)
dev.off()

fit_cov_poly <- ppm(sample ~ polynom(roads, 3) + polynom(camps, 3) +
                      polynom(picnic, 3) + polynom(gates, 3) +
                      polynom(border, 3) + polynom(dams, 3) +
                      polynom(water, 3), covariates = covariates)
fit_back_cov <- step(fit_cov, direction = "backward")
fit_back_cov_poly <- step(fit_cov_poly, direction = "backward")
fit_forw_cov <- step(fit_csr, fit_cov$trend, direction = "forward")
fit_forw_cov_poly <- step(fit_csr, fit_cov_poly$trend, direction = "forward")

fit_cov_inter <- ppm(sample ~ .^2, covariates = covariates)
fit_back_cov_inter <- step(fit_cov_inter, direction = "backward")
fit_forw_cov_inter <- step(fit_csr, fit_cov_inter$trend, direction = "forward")

fit_cov_poly_inter <- update(fit_cov_poly, ~ .^2)
fit_back_cov_poly_inter <- step(fit_cov_poly_inter, direction = "backward")
fit_forw_cov_poly_inter <- step(fit_csr, fit_cov_poly_inter$trend,
                                direction = "forward")

fit_all <- anylist(fit_csr, fit_cov, fit_z1, fit_z2, fit_lrt,
                  fit_cov_poly, fit_back_cov, fit_back_cov_poly,
                  fit_forw_cov, fit_forw_cov_poly,
                  fit_cov_inter, fit_back_cov_inter, fit_forw_cov_inter,
                  fit_cov_poly_inter, fit_back_cov_poly_inter,
                  fit_forw_cov_poly_inter)
fit_all <- c(fit_all, fit_polys)
fit_all_names <- c("CSR", "Full covariate", "Z1", "Z2", "LRT",
                  "Full covariate poly",
                  "Backward covariate", "Backward covariate poly",
                  "Forward covariate", "Forward covariate poly",
                  "Full covariate interact",
                  "Backward covariate interact",
                  "Forward covariate interact",
                  "Full covariate poly interact",
                  "Backward covariate poly interact",
                  "Forward covariate poly interact",
                  paste(names(covariates), "poly"))
gof <- function(x) {
  if (length(coef(x)) <= 10) {
    qt <- quadrat.test(x, nx = 4, ny = 3)
    ret <- c(qt$parameter, qt$p.value)
  } else if (length(coef(x)) <= 24) {
    qt <- quadrat.test(x, nx = 6, ny = 4)
    ret <- c(qt$parameter, qt$p.value)
  } else if (length(coef(x)) <= 30) {
    qt <- quadrat.test(x, nx = 6, ny = 5)
    ret <- c(qt$parameter, qt$p.value)
  }  else ret <- c(NA, NA)
  return(ret)
}
fit_all_df <- data.frame(p = sapply(fit_all, function(x) {length(coef(x))}),
              AIC = sapply(fit_all, AIC),
              MAE = sapply(fit_all, function(x) sum(abs(residuals(x)$val)) / n),
              Chi2_df = sapply(fit_all, function(x) gof(x)[1]),
              Chi2_pval = sapply(fit_all, function(x) gof(x)[2]),
              row.names = fit_all_names)
fit_all_df <- rbind(fit_all_df, 
              cov_df[, c("p", "AIC", "MAE", "Chi2_df", "Chi2_pval")])
round(fit_all_df[order(fit_all_df$AIC), ], 3)

# plot intensity and pp
xlab <- "Easting distance (km)"
ylab <- "Northing distance (km)"
pdf(paste0(path, "model_given_intensity.pdf"), width = 5, height = 3)
par(mfrow = c(1, 1), mar = c(2, 2, 0, 2), oma = c(1, 1, 0, 1))
plot(reflect(lambda), main = "", axes = TRUE, cex.axis = 0.8,
    ribsep = 0.05, ribside = "right")
plot(reflect(sample), add = TRUE, cex = 0.75, pch = 16, cols = rgb(0, 0, 0, 0.6))
mtext("Intensity", side = 4, line = -0.5, outer = TRUE)
mtext(xlab, side = 1, line = 0, outer = TRUE)
mtext(ylab, side = 2, line = 0, outer = TRUE)
dev.off()

# plot fitted models
fit_adj <- predict(fit_cov, type = "intensity") / p
fit_se <- predict(fit_cov, type = "intensity", se = TRUE)$se
pdf(paste0(path, "model_fitted_intensity_cov.pdf"), width = 7, height = 4)
par(mfrow = c(1, 2), mar = c(2, 2, 2, 0), oma = c(1, 1, 1, 0))
plot(reflect(fit_adj), main = "", axes = TRUE, cex.axis = 0.8,
    ribsep = 0.05, ribside = "top")
mtext("Fitted Intensity", side = 3, line = 1, cex = 0.8)
plot(reflect(fit_se), main = "", axes = TRUE, cex.axis = 0.8,
    ribsep = 0.05, ribside = "top")
mtext("SE of Fitted Intensity", side = 3, line = 1, cex = 0.8)
mtext(xlab, side = 1, line = -1, outer = TRUE)
mtext(ylab, side = 2, line = 0, outer = TRUE)
dev.off()

# plot top 4 fitted models
best <- anylist(fit_back_cov_poly, fit_forw_cov_poly,
                fit_back_cov, fit_forw_cov_inter)
panel <- c("(a)", "(b)", "(c)", "(d)")
pdf(paste0(path, "model_fitted_intensity_best.pdf"), width = 7, height = 4.5)
par(mfrow = c(2, 2), mar = c(2, 2, 1, 3), oma = c(1, 1, 0, 1), cex.main = 0.8)
for (i in seq_along(best)) {
  fit <- predict(best[[i]], type = "intensity") / p
  plot(reflect(fit), main = "", axes = TRUE, cex.axis = 0.8, ribsep = 0.05)
    mtext("Fitted Intensity", side = 4, line = 1.5, cex = 0.7)
}
mtext(xlab, side = 1, line = 0, outer = TRUE)
mtext(ylab, side = 2, line = 0, outer = TRUE)
dev.off()

save.image(file = paste0(path, "knp_pp_model.RData"))