library(sf)
library(spatstat)
library(ggplot2)
theme_set(theme_bw())
library(patchwork)
library(reshape2)

path <- "C:/Users/lisak/OneDrive/PhD2.0/Coding/Simulations/spatstat/final_ex1/"
# load(paste0(path, "knp_poach.RData"))
load(paste0(path, "knp_sim_sample.RData"))

# plot gps grid
pdf(paste0(path, "grid_gps.pdf"), width = 7, height = 12)
plot(grid_gps, axes = TRUE)
dev.off()

# calculate window from projected grid and sample window
w <- as.owin(grid_proj) # full window
w <- rescale(w, 1000, "km")
b <- owin(c(-68.33145, -35.92676), c(2694.3555, 2715.5711))
b <- w[b] # smaller sampling window
notb <- setminus.owin(w, b) # complement of b in w
w_area <- area.owin(w)
b_area <- area.owin(b)

## simulation study - simulate data different ways, vary sample size

# generate point patterns
set.seed(40)
iter <- 1000
sim_data <- c("CSR", "Inhomogeneous Poisson", "Sampling Window",
              "p-Thinning", "p(u)-Thinning", "P(u)-Thinning")
sample_size <- c(50, 100, 200, 300, 400, 500, 1000, 5000)
results <- data.frame()

for (i in 1:iter) {
  print(paste("iteration: ", i, sep = ""))
  for (j in seq_along(sim_data)) {
    print(paste("data simulation method: ", j, sep = ""))
    for (n in seq_along(sample_size)) {
        if (j == 1) { # csr
            lambda <- sample_size[n] / w_area
            pp <- rpoispp(lambda, win = w)
        }
        if (j == 2) { # inhomogeneous
            lambda_u <- sample_size[n] * poach_prob
            pp <- rpoispp(lambda_u)
        }
        if (j == 3) { # window
            p <- 0.8 * b_area / w_area
            n_large <- round(sample_size[n] / p)
            lambda_u <- n_large * poach_prob
            pp_large <- rpoispp(lambda_u)
            pp <- pp_large[b]
        }
        if (j == 4) { # p-thin
            p <- 0.5
            n_large <- round(sample_size[n] / p)
            lambda_u <- n_large * poach_prob
            pp_large <- rpoispp(lambda_u)
            pp <- rthin(pp_large, p)
        }
        if (j == 5) { # pu-thin
            p_u <- 0.5 * poach_prob /  mean(poach_prob)
            p_u <- p_u / max(p_u)
            n_large <- round(sample_size[n] / mean(p_u))
            lambda <- n_large / w_area
            pp_large <- rpoispp(lambda, win = w)
            pp <- rthin(pp_large, p_u)
        }
        if (j == 6) { # Pu-thin
            p_u <- rnoise(runif, poach_prob, max = 0.5)
            p_u <- Smooth(p_u, sigma = 1, normalise = TRUE, bleed = FALSE)
            p_u <- p_u + 0.25 * poach_prob /  mean(poach_prob)
            p_u <- p_u / max(p_u)
            n_large <- round(sample_size[n] / mean(p_u))
            lambda <- n_large / w_area
            pp_large <- rpoispp(lambda, win = w)
            pp <- rthin(pp_large, p_u)
        }
        fit <- ppm(pp ~ ., covariates = covariates)
        residual <- residuals(fit)
        res <- residual$val
        data_points <- residual$discrete
        res_data <- res[data_points]
        res_diffuse <- res[!data_points]
        df_iter <- data.frame(iter = i,
                                sim_data = sim_data[j],
                                sample_size = sample_size[n],
                                n = npoints(pp),
                                intensity = intensity(pp),
                                p = length(coef(fit)),
                                res_len = length(res),
                                res_sum = sum(res),
                                res_mean = mean(res),
                                res_var = var(res),
                                res_abs_sum = sum(abs(res)),
                                res_sum_sq = sum(res^2),
                                MAE = sum(abs(res)) / length(res),
                                MSE = sum(res^2) / length(res),
                                RMSE = sqrt(sum(res^2) / length(res)),
                                res_data_len = length(res_data),
                                res_data_sum = sum(res_data),
                                res_data_mean = mean(res_data),
                                res_data_var = var(res_data),
                                res_data_abs_sum = sum(abs(res_data)),
                                res_data_sum_sq = sum(res_data^2),
                                MAE_data = sum(abs(res_data)) / length(res_data),
                                MSE_data = sum(res_data^2) / length(res_data),
                                RMSE_data = sqrt(sum(res_data^2) / length(res_data)),
                                logLik = logLik(fit),
                                deviance = deviance(fit),
                                AIC = AIC(fit),
                                BIC = BIC(fit)
        )
        results <- rbind(results, df_iter)
    }
  }
}

# remove simulation variables
rm(iter, i, j, n, p, p_u, n_large, lambda, lambda_u, pp, pp_large, df_iter)

# factor variables
results$sim_data <- factor(results$sim_data, levels = sim_data)
results$sample_size <- factor(results$sample_size, levels = sample_size)


# plots of mean absolute error
plot_measure <- function(df, measure_col) {
  measure_name <- names(df)[measure_col]
  file_name <- paste0("sim_", measure_name, ".pdf")
  measure <- df[, measure_col]
  box <- ggplot(df, aes(y = sample_size, x = measure)) +
                geom_boxplot() +
                facet_wrap(~sim_data, ncol = 2) +
                labs(y = "Sample size") +
                theme(axis.title.x = element_blank())
  density <- ggplot(df, aes(measure, 
                color = sample_size)) +
                  geom_density(alpha = 0.3) +
                  facet_wrap(~sim_data, ncol = 2) +
                  labs(x = measure_name, y = "Density") +
                  scale_color_discrete(name = "Sample size",
                    guide = guide_legend(nrow = 2)) +
                  theme(legend.position = c(0.83, 0.9),
                    legend.title = element_blank(),
                    legend.text = element_text(size = 10),
                    legend.key.size = unit(0.2, "cm"),
                    legend.background = element_blank(),
                    legend.box.background = element_rect(color = "black"))
  measure_plot <- box / density
  ggsave(filename = file_name, plot = measure_plot,
          device = "pdf", path = path, width = 7, height = 9,
          units = "in", dpi = 500, limitsize = TRUE)
}
plot_measure(results, 13)
plot_measure(results, 14)
plot_measure(results, 15)
plot_measure(results, 22)
plot_measure(results, 23)
plot_measure(results, 24)
plot_measure(results, 27)
plot_measure(results, 28)

results$MAE_data_2 <- results$res_data_abs_sum / results$res_len
results$MAE_n_denom <- results$res_abs_sum / results$n

box1 <- ggplot(results, aes(y = sample_size, x = MAE_n_denom)) +
                geom_boxplot() +
                facet_wrap(~sim_data, ncol = 2) +
                labs(y = "Sample size", 
                  title = "(a) Sum over all residuals, divide by n") +
                theme(axis.title.x = element_blank())
box2 <- ggplot(results, aes(y = sample_size, x = MAE_data)) +
                geom_boxplot() +
                facet_wrap(~sim_data, ncol = 2) +
                labs(title = "(b) Sum over point residuals, divide by n") +
                theme(axis.title.x = element_blank()) +
                theme(axis.title.y = element_blank())
box3 <- ggplot(results, aes(y = sample_size, x = MAE)) +
                geom_boxplot() +
                facet_wrap(~sim_data, ncol = 2) +
                labs(y = "Sample size", x = "Mean absolute error",
                  title = "(c) Sum over all residuals, divide by residuals length")
box4 <- ggplot(results, aes(y = sample_size, x = MAE_data_2)) +
                geom_boxplot() +
                facet_wrap(~sim_data, ncol = 2) +
                labs(x = "Mean absolute error",
                  title = "(d) Sum over point residuals, divide by residuals length")+
                theme(axis.title.y = element_blank())
measure_plot <- (box1 + box2) / (box3 + box4)
ggsave(filename = "sim_mae_calc.pdf", plot = measure_plot,
          device = "pdf", path = path, width = 12, height = 9,
          units = "in", dpi = 500, limitsize = TRUE)

# df for sample size 200
results_fit_200 <- results_fit[results_fit$sample_size == 200, ]
results_200 <- results_df[results_df$sample_size == 200, ]
# remove large object
rm(results_fit)

# plots of number of points
sim_labs <- c("CSR", "INP", "WIN", "p", "p(u)", "P(u)")
names(sim_labs) <- sim_data
s_npoints_box <- ggplot(results_200, aes(y = sim_data, x = npoints)) +
              geom_boxplot() +
              labs(y = "Point Process") +
              theme(axis.title.x = element_blank()) +
                xlim(150, 255) +
              scale_y_discrete(labels = sim_labs)
agg_npoint <- aggregate(npoints ~ sim_data, data = results_200, FUN = mean)
tmp <- merge(results_200, agg_npoint, by = "sim_data")
s_npoints_dens <- ggplot(tmp) +
                geom_line(aes(x = npoints.x,
                    y = dpois(npoints.x, npoints.y)),
                    col = "tomato") +
                geom_density(aes(npoints.x), alpha = 0.8) +
                facet_wrap(~sim_data, ncol = 1, scales = "free") +
                xlim(150, 255) +
                labs(x = "Number of Points", y = "Density")

# plots of intensity
tmp <- results_200[results_200$sim_data != "Sampling Window", ]
s_int_box <- ggplot(tmp, aes(y = sim_data, x = intensity),
                    subset = .(sim_data != "Sampling Window")) +
              geom_boxplot() +
              theme(axis.title.x = element_blank(),
                axis.title.y = element_blank()) +
                xlim(0.065, 0.115) +
              scale_y_discrete(labels = sim_labs[-3])
agg_int <- aggregate(intensity ~ sim_data, data = results_200,
    FUN = function(x) c(mean = mean(x), sd = sd(x)))
tmp <- merge(results_200, agg_int, by = "sim_data")
tmp$x_min <- ifelse(tmp$sim_data == "Sampling Window", 0.25, 0.065)
tmp$x_max <- ifelse(tmp$sim_data == "Sampling Window", 0.425, 0.115)
s_int_dens <- ggplot(tmp) +
                facet_wrap(~sim_data, ncol = 1, scales = "free") +
                geom_line(aes(x = intensity.x,
                    y = dnorm(intensity.x, intensity.y[, 1], intensity.y[, 2])),
                    col = "turquoise") +
                geom_density(aes(intensity.x), alpha = 0.8) +
                geom_blank(aes(x = x_min)) +
                geom_blank(aes(x = x_max)) +
                labs(x = "Mean Intensity", y = "")

# plot npoints and intensity side by side
s_distr <- (s_npoints_box + s_int_box) / (s_npoints_dens + s_int_dens) +
              plot_layout(heights = c(1.5, 8))
ggsave(filename = "sim_sample_distr.pdf", plot = s_distr,
        device = "pdf", path = path, width = 7, height = 10,
        units = "in", dpi = 500, limitsize = TRUE)

# plot mae for csr and cov
tmp <- melt(results_200, id.vars = c("sim_data", "sample_size"),
            measure.vars = c("mae_csr", "mae_cov"))
sim_labs_fill <- c("CSR", "Covariate")
names(sim_labs_fill) <- c("mae_csr", "mae_cov")
sim_labs_x <- c("CSR", "INP", "WIN", "p", "p(u)", "P(u)")
names(sim_labs_x) <- sim_data
s_mae_ss <- ggplot(tmp, aes(x = sim_data, y = value, fill = variable)) +
              geom_boxplot() +
              labs(x = "Point process", y = "Mean absolute error") +
                theme(legend.position = c(0.1, 0.12),
                  legend.background = element_blank(),
                  legend.box.background = element_rect(colour = "black")) +
                scale_fill_discrete(name = "Model", labels = sim_labs_fill,
                  guide = guide_legend(ncol = 1)) +
              scale_x_discrete(labels = sim_labs_x)
ggsave(filename = "sim_sample_mae_csr.pdf", plot = s_mae_ss,
        device = "pdf", path = path, width = 7, height = 5,
        units = "in", dpi = 500, limitsize = TRUE)

# plot variables
process <- c("Sampling Window", "p-Thinning")
tmp <- results_fit_200[results_fit_200$sim_data %in% process, ]
tmp2 <- as.data.frame(with(tmp, summary(fit_cov)$coef[, 6]))
names <- c("Intercept", names(covariates))
colnames(tmp2) <- c("Intercept", names(covariates))
results_var <- as.data.frame(tmp[, c("sim_data", "sample_size")])
results_var <- cbind(results_var, tmp2)
tmp3 <- melt(results_var, id.vars = c("sim_data", "sample_size"),
            measure.vars = names)

s_var_plot <- ggplot(tmp3, aes(x = variable, y = value, fill = sim_data)) +
                geom_boxplot() +
                annotate("rect", xmin = 0, xmax = 9, ymin = -2,
                         ymax = 2, alpha = 0.5, fill = "grey") +
                labs(x = "Covariate",
                  y = "Wald statistic to test for zero coefficient") +
                theme(legend.position = c(0.87, 0.12),
                  legend.background = element_blank(),
                  legend.box.background = element_rect(colour = "black")) +
                scale_fill_discrete(name = "Process", labels = sim_labs_fill,
                  guide = guide_legend(ncol = 1)) +
              scale_x_discrete(labels = sim_labs_x)
ggsave(filename = "sim_sample_variables.pdf", plot = s_var_plot,
              device = "pdf", path = path, width = 7, height = 5,
              units = "in", dpi = 500, limitsize = TRUE)

# best pp
by(results_200, results_200$sim_data,
    FUN = function(x) rownames(x)[which.min(x$mae_cov)])
pp_win <- results["123827", "pp", drop = TRUE]
pp_pthin <- results["139579", "pp", drop = TRUE]

# remove objects
rm(agg_int, agg_npoint, results_200, tmp, tmp2, tmp3, names, notb, process,
  s_gof, s_gof_cov, s_gof_csr, s_int, s_int_box, s_int_dens, s_mae, s_mae_box,
  s_mae_dens, s_mae_ss, s_npoints, s_npoints_box, s_npoints_dens, s_var_plot,
  sample_size, sim_data, sim_labs, sim_labs_fill, sim_labs_x, results_fit_200)

save.image(file = paste0(path, "knp_sim_sample.RData"))

library("writexl")
write_xlsx(results, "spat_sim.xlsx")