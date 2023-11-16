library(sf)
library(spatstat)
library(ggplot2)
theme_set(theme_bw())
library(patchwork)
library(reshape2)
library(raster)
library(spdep)

# Load data
path <- "C:/Users/lisak/OneDrive/PhD2.0/Coding/Simulations/spatstat/final_ex1/"
load(paste0(path, "knp_poach.RData"))

# colour blind palette
cbPalette <- c("#999999", "#E69F00", "#56B4E9", "#009E73",
                "#F0E442", "#0072B2", "#D55E00", "#CC79A7")


sim_dummy <- function(iter, seed, prob, covariates, data_gen_index) {
  # simulation parameters
  set.seed(seed)
  #dummy_pt_prop <- c(2, 4, 6, 8, 10)
  dummy_pt_prop <- c(10)
  sample_size <- c(50, 100, 200, 300, 400, 500, 1000, 5000)
  data_gen <- c("CSR", "IP", "WIN", "pTHIN", "puTHIN", "PuTHIN")
  names <- c("Roads", "Camps", "Picnic", "Gates", "Border", "Dams", "Water")

  # calculate window from projected grid and sample window
  w <- as.owin(grid_proj) # full window
  w <- rescale(w, 1000, "km")
  b <- owin(c(-68.33145, -35.92676), c(2694.3555, 2715.5711))
  b <- w[b] # smaller sampling window
  notb <- setminus.owin(w, b) # complement of b in w
  w_area <- area.owin(w)
  b_area <- area.owin(b)

  # simulation results
  results <- data.frame()
  results_dummy <- data.frame()
  variables <- data.frame()
  variables_dummy <- data.frame()

  for (i in 1:iter) {
    print(paste("iteration: ", i, sep = ""))
    for (dummy_prop in dummy_pt_prop) {
        print(paste("dummy prop: ", dummy_prop, sep = ""))
        for (n in seq_along(sample_size)) {
            print(paste("sample size: ", sample_size[n], sep = ""))

        # sample size
        mu <- sample_size[n]

        # intensity
        lambda <- mu * prob
        ave_lambda <- mean(lambda)

        # point pattern
            if (data_gen_index == 1) { # csr
                lambda <- mu / w_area
                pp <- rpoispp(lambda, win = w)
            }
            if (data_gen_index == 2) { # inhomogeneous
                lambda_u <- mu * prob
                pp <- rpoispp(lambda_u)
            }
            if (data_gen_index == 3) { # window
                p <- 0.8 * b_area / w_area
                n_large <- round(mu / p)
                lambda_u <- n_large * prob
                pp_large <- rpoispp(lambda_u)
                pp <- pp_large[b]
            }
            if (data_gen_index == 4) { # p-thin
                p <- 0.5
                n_large <- round(mu / p)
                lambda_u <- n_large * prob
                pp_large <- rpoispp(lambda_u)
                pp <- rthin(pp_large, p)
            }
            if (data_gen_index == 5) { # pu-thin
                p_u <- 0.5 * prob /  mean(prob)
                p_u <- p_u / max(p_u)
                n_large <- round(mu / mean(p_u))
                lambda <- n_large / w_area
                pp_large <- rpoispp(lambda, win = w)
                pp <- rthin(pp_large, p_u)
            }
            if (data_gen_index == 6) { # Pu-thin
                p_u <- rnoise(runif, prob, max = 0.5)
                p_u <- Smooth(p_u, sigma = 1, normalise = TRUE, bleed = FALSE)
                p_u <- p_u + 0.25 * prob /  mean(prob)
                p_u <- p_u / max(p_u)
                n_large <- round(mu / mean(p_u))
                lambda <- n_large / w_area
                pp_large <- rpoispp(lambda, win = w)
                pp <- rthin(pp_large, p_u)
            }
        n_pts <- npoints(pp)
        ave_intensity <- intensity(pp)

        # dummy point pattern
        dummy_lambda <- dummy_prop * ave_intensity
        if (data_gen_index == 3) {
            dummy_win <- b
        } else {
            dummy_win <- w
        }
        dummy_pp <- rpoispp(dummy_lambda, win = dummy_win)
        d <- npoints(dummy_pp)

        # default fit
        fit <- ppm(pp ~ ., covariates = covariates)

        # fit with dummy pp
        dummy_quad <- quadscheme(pp, dummy_pp)
        dummy_fit <- ppm(dummy_quad ~ ., covariates = covariates)

        # residuals
        residual <- residuals(fit)
        res <- residual$val
        diff_points <- length(res[!residual$discrete])
        mae <- mean(abs(res))

        # dummy residuals
        residuals_dummy <- residuals(dummy_fit)
        dummy_res <- residuals_dummy$val
        dummy_mae <- mean(abs(dummy_res))

        # predictions
        fit_intensity <- predict(fit, locations = prob,
            covariates = covariates, type = "intensity")
        fit_ave_lambda <- mean(fit_intensity)
        fit_prob <- fit_intensity / mu

        dummy_fit_intensity <- predict(dummy_fit, locations = prob,
            covariates = covariates, type = "intensity")
        dummy_fit_ave_lambda <- mean(dummy_fit_intensity)
        dummy_fit_prob <- dummy_fit_intensity / mu

        # CSR model
        csr_fit <- ppm(pp ~ 1)
        csr_res <- residuals(csr_fit)$val
        csr_mae <- mean(abs(csr_res))
        dummy_csr_fit <- ppm(dummy_quad ~ 1)
        dummy_csr_res <- residuals(dummy_csr_fit)$val
        dummy_csr_mae <- mean(abs(dummy_csr_res))

        # lambda and prob error
        lambda_err <- lambda - fit_intensity
        lambda_mae <- mean(abs(lambda_err))
        prob_err <- prob - fit_prob
        prob_mae <- mean(abs(prob_err))

        # dummy lambda and prob error
        dummy_lambda_err <- lambda - dummy_fit_intensity
        dummy_lambda_mae <- mean(abs(dummy_lambda_err))
        dummy_prob_err <- prob - dummy_fit_prob
        dummy_prob_mae <- mean(abs(dummy_prob_err))

        # AIC
        fit_aic <- AIC(fit)
        dummy_aic <- AIC(dummy_fit)
        fit_csr_aic <- AIC(csr_fit)
        dummy_csr_aic <- AIC(dummy_csr_fit)

        res_iter <- data.frame(iter = i, dummy_gen = "Default",
                                data_gen = data_gen[data_gen_index],
                                dummy_prop = dummy_prop,
                                sample_size = mu,
                                ave_lambda = ave_lambda,
                                pp_ave_lambda = ave_intensity,
                                fit_ave_lambda = fit_ave_lambda,
                                data_points = n_pts,
                                dummy_points = diff_points,
                                mae = mae,
                                lambda_mae = lambda_mae,
                                prob_mae = prob_mae,
                                csr_mae = csr_mae,
                                AIC = fit_aic,
                                csr_aic = fit_csr_aic)
        results <- rbind(results, res_iter)
        res_dummy_iter <- data.frame(iter = i, dummy_gen = "CSR",
                                data_gen = data_gen[data_gen_index],
                                dummy_prop = dummy_prop,
                                sample_size = mu,
                                ave_lambda = ave_lambda,
                                pp_ave_lambda = ave_intensity,
                                fit_ave_lambda = dummy_fit_ave_lambda,
                                data_points = n_pts,
                                dummy_points = d,
                                mae = dummy_mae,
                                lambda_mae = dummy_lambda_mae,
                                prob_mae = dummy_prob_mae,
                                csr_mae = dummy_csr_mae,
                                AIC = dummy_aic,
                                csr_aic = dummy_csr_aic)
        results_dummy <- rbind(results_dummy, res_dummy_iter)

        coef_fit <- summary(fit)
        var_iter <- data.frame(iter = i, dummy_gen = "Default",
                                data_gen = data_gen[data_gen_index],
                                dummy_prop = dummy_prop,
                                sample_size = mu,
                                term = c("Intercept", names),
                                coef = coef_fit$coef[, 1],
                                se = coef_fit$coef[, 2],
                                wald = coef_fit$coef[, 6])
        variables <- rbind(variables, var_iter)
        dummy_coef_fit <- summary(dummy_fit)
        var_dummy_iter <- data.frame(iter = i, dummy_gen = "CSR",
                                data_gen = data_gen[data_gen_index],
                                dummy_prop = dummy_prop,
                                sample_size = mu,
                                term = c("Intercept", names),
                                coef = dummy_coef_fit$coef[, 1],
                                se = dummy_coef_fit$coef[, 2],
                                wald = dummy_coef_fit$coef[, 6])
        variables_dummy <- rbind(variables_dummy, var_dummy_iter)
        }
    }
  }
  results <- rbind(results, results_dummy)
  variables <- rbind(variables, variables_dummy)
  return(list(results, variables))
}

# for "parallel" runs
sim1 <- sim_dummy(1000, 1000, poach_prob, covariates, data_gen_index = 1)
save.image(file = paste0(path, "sim_dummy_1.RData"))

sim2 <- sim_dummy(1000, 100000, poach_prob, covariates, data_gen_index = 2)
save.image(file = paste0(path, "sim_dummy_2.RData"))

sim3 <- sim_dummy(1000, 1000000, poach_prob, covariates, data_gen_index = 3)
save.image(file = paste0(path, "sim_dummy_3.RData"))

sim4 <- sim_dummy(1000, 10000000, poach_prob, covariates, data_gen_index = 4)
save.image(file = paste0(path, "sim_dummy_4.RData"))

sim5 <- sim_dummy(1000, 100000000, poach_prob, covariates, data_gen_index = 5)
save.image(file = paste0(path, "sim_dummy_5.RData"))

sim6 <- sim_dummy(1000, 1000000000, poach_prob, covariates, data_gen_index = 6)
save.image(file = paste0(path, "sim_dummy_6.RData"))

load(paste0(path, "sim_dummy_1.RData"))
load(paste0(path, "sim_dummy_2.RData"))
load(paste0(path, "sim_dummy_3.RData"))
load(paste0(path, "sim_dummy_4.RData"))
load(paste0(path, "sim_dummy_5.RData"))
load(paste0(path, "sim_dummy_6.RData"))

results <- data.frame()
variables <- data.frame()
sim <- list(sim1, sim2, sim3, sim4, sim5, sim6)
for (i in seq_along(sim)) {
    results <- rbind(results, sim[[i]][[1]])
    variables <- rbind(variables, sim[[i]][[2]])
}

# factor columns
dummy_pt_prop <- c(2, 4, 6, 8, 10)
sample_size <- c(50, 100, 200, 300, 400, 500, 1000, 5000)
data_gen <- c("CSR", "IP", "WIN", "pTHIN", "puTHIN", "PuTHIN")
names <- c("Roads", "Camps", "Picnic", "Gates", "Border", "Dams", "Water")

results$dummy_gen <- factor(results$dummy_gen, levels = c("Default", "CSR"))
results$data_gen <- factor(results$data_gen, levels = data_gen)
results$dummy_prop <- factor(results$dummy_prop, levels = dummy_pt_prop)
results$sample_size <- factor(results$sample_size, levels = sample_size)

variables$dummy_gen <- factor(variables$dummy_gen, levels = c("Default", "CSR"))
variables$data_gen <- factor(variables$data_gen, levels = data_gen)
variables$dummy_prop <- factor(variables$dummy_prop, levels = dummy_pt_prop)
variables$sample_size <- factor(variables$sample_size, levels = sample_size)
variables$term <- factor(variables$term, levels = c("Intercept", names))

rm(sim1, sim2, sim3, sim4, sim5, sim6, sim, i)
save.image(file = paste0(path, "sim_dummy.RData"))

# boxplots of MAE and dummy points for default fit
df <- results[results$dummy_gen == "Default" & results$dummy_prop == 2, ]
def_col <- cbPalette[3]
box_mae <- ggplot(df, aes(y = sample_size, x = mae)) +
                geom_boxplot(fill = def_col) +
                facet_wrap(~data_gen, ncol = 1) +
                labs(y = "Sample size", x = "Mean absolute error") +
                theme(strip.text = element_blank())
box_dummy <- ggplot(df, aes(y = sample_size, x = dummy_points)) +
                geom_boxplot(fill = def_col) +
                facet_wrap(~data_gen, ncol = 1) +
                labs(x = "Number of dummy points") +
                theme(axis.title.y = element_blank(),
                        axis.text.y = element_blank(),
                        axis.ticks.y = element_blank(),
                        strip.text = element_blank())
box_mae_dummy <- box_mae + box_dummy + plot_layout(widths = c(1.5, 1))
ggsave(filename = "sim_mae.pdf", plot = box_mae_dummy,
          device = "pdf", path = path, width = 7, height = 9,
          units = "in", dpi = 600, limitsize = TRUE)

# boxplots of MAE and dummy points for CSR dummy point fit
df <- results[results$dummy_gen == "CSR" & results$dummy_prop == 10, ]
dummy_col <- cbPalette[7]
box_mae <- ggplot(df, aes(y = sample_size, x = mae)) +
                geom_boxplot(fill = dummy_col) +
                facet_wrap(~data_gen, ncol = 1) +
                labs(y = "Sample size", x = "Mean absolute error") +
                theme(strip.text = element_blank())
box_dummy <- ggplot(df, aes(y = sample_size, x = dummy_points)) +
                geom_boxplot(fill = dummy_col) +
                facet_wrap(~data_gen, ncol = 1) +
                labs(x = "Number of dummy points") +
                theme(axis.title.y = element_blank(),
                        axis.text.y = element_blank(),
                        axis.ticks.y = element_blank(),
                        strip.text = element_blank())
box_mae_dummy <- box_mae + box_dummy + plot_layout(widths = c(1.5, 1))
ggsave(filename = "sim_dummy_mae_prop10.pdf", plot = box_mae_dummy,
          device = "pdf", path = path, width = 7, height = 9,
          units = "in", dpi = 600, limitsize = TRUE)

# boxplots of MAE for varying dummy proportions
df <- results[results$dummy_gen == "CSR" & results$data_gen == "IP", ]
box_mae_prop <- ggplot(df, aes(y = dummy_prop, x = mae)) +
                geom_boxplot(fill = dummy_col) +
                facet_wrap(~sample_size, ncol = 2) +
                labs(y = "Proportion of dummy points",
                        x = "Mean absolute error") +
                theme(strip.text = element_blank())
ggsave(filename = "sim_ip_prop.pdf", plot = box_mae_prop,
          device = "pdf", path = path, width = 7, height = 6,
          units = "in", dpi = 600, limitsize = TRUE)

# boxplots of coefficients for default and dummy fit
df <- variables[variables$dummy_prop == 10 &
                        variables$data_gen == "IP", ]
dummy_palette <- c(def_col, dummy_col)
sim_labs <- c("Default dummy points", "CSR dummy points")
box_coef <- ggplot(df, aes(y = sample_size, x = coef,
                    fill = dummy_gen)) +
              geom_boxplot() +
              facet_wrap(~term, ncol = 2, scales = "free") +
              labs(y = "Sample size", x = "Coefficient") +
              scale_fill_manual(values = dummy_palette, name = "",
                                labels = sim_labs) +
              theme(legend.position = "top",
                    strip.text = element_blank())
ggsave(filename = "sim_ip_coef.pdf", plot = box_coef,
        device = "pdf", path = path, width = 7, height = 9,
        units = "in", dpi = 600, limitsize = TRUE)

# boxplots of prob mae for default and dummy fit
options(scipen=5)
df <- results[results$dummy_prop == 10, ]
dummy_palette <- c(def_col, dummy_col)
sim_labs <- c("Default dummy points", "CSR dummy points")

box_prob <- ggplot(df, aes(y = sample_size, x = prob_mae,
                        fill = dummy_gen)) +
                geom_boxplot() +
                labs(y = "Sample size",
                        x = "Mean absolute error of probability") +
              facet_wrap(~data_gen, ncol = 2, scales = "free") +
              scale_fill_manual(values = dummy_palette, name = "",
                                labels = sim_labs) +
              theme(legend.position = "top",
                    strip.text = element_blank())
ggsave(filename = "sim_dummy_prob_error.pdf", plot = box_prob,
          device = "pdf", path = path, width = 7, height = 7,
          units = "in", dpi = 600, limitsize = TRUE)

# Sample size 200, distributions
df <- results[results$dummy_gen == "Default" & results$dummy_prop == 2, ]
results_200 <- df[df$sample_size == 200, ]

# plots of number of points
sim_labs <- c("CSR", "IP", "WIN", "pTHIN", "p(u)THIN", "P(u)THIN")
names(sim_labs) <- data_gen
s_n_box <- ggplot(results_200, aes(y = data_gen, x = data_points)) +
              geom_boxplot() +
              labs(y = "Point Process") +
              theme(axis.title.x = element_blank(),
                    strip.text = element_blank()) +
              scale_y_discrete(labels = sim_labs, limits = rev) +
              xlim(140, 260)
agg_npoint <- aggregate(data_points ~ data_gen, data = results_200, FUN = mean)
tmp <- merge(results_200, agg_npoint, by = "data_gen")
s_n_dens <- ggplot(tmp) +
                geom_line(aes(x = data_points.x,
                    y = dpois(data_points.x, data_points.y)),
                    col = cbPalette[8], size = 0.7, alpha = 0.9) +
                geom_density(aes(data_points.x), alpha = 0.8) +
                facet_wrap(~data_gen, ncol = 1, scales = "free") +
                labs(x = "Number of Points", y = "Density") +
                theme(strip.text = element_blank())
                xlim(140, 260)

# plots of intensity
tmp <- results_200[results_200$data_gen != "WIN", ]
s_int_box <- ggplot(tmp, aes(y = data_gen, x = pp_ave_lambda),
                    subset = .(data_gen != "WIN")) +
              geom_boxplot() +
              theme(axis.title.x = element_blank(),
                axis.title.y = element_blank(),
                strip.text = element_blank()) +
              scale_y_discrete(labels = sim_labs[-3], limits = rev) +
              xlim(0.06, 0.12)
agg_int <- aggregate(pp_ave_lambda ~ data_gen, data = results_200,
    FUN = function(x) c(mean = mean(x), sd = sd(x)))
tmp <- merge(results_200, agg_int, by = "data_gen")
tmp$x_min <- ifelse(tmp$data_gen == "WIN", 0.25, 0.06)
tmp$x_max <- ifelse(tmp$data_gen == "WIN", 0.4125, 0.12)
s_int_dens <- ggplot(tmp) +
                facet_wrap(~data_gen, ncol = 1, scales = "free") +
                geom_line(aes(x = pp_ave_lambda.x,
                    y = dnorm(pp_ave_lambda.x, pp_ave_lambda.y[, 1],
                                    pp_ave_lambda.y[, 2])),
                    col = cbPalette[4], size = 0.7, alpha = 0.9) +
                geom_density(aes(pp_ave_lambda.x), alpha = 0.8) +
                geom_blank(aes(x = x_min)) +
                geom_blank(aes(x = x_max)) +
                labs(x = "Mean Intensity", y = "") +
                theme(strip.text = element_blank())

# plot points and intensity side by side
s_distr <- (s_n_box + s_int_box) / (s_n_dens + s_int_dens) +
              plot_layout(heights = c(1.5, 8))
ggsave(filename = "sim_200_distr.pdf", plot = s_distr,
        device = "pdf", path = path, width = 7, height = 9,
        units = "in", dpi = 600, limitsize = TRUE)

# plot mae and for csr and cov
df <- results[results$dummy_prop == 10 & results$dummy_gen == "CSR", ]
results_200 <- df[df$sample_size == 200, ]

tmp_mae <- melt(results_200, id.vars = c("data_gen"),
            measure.vars = c("csr_mae", "mae"))
tmp_mae$model <- ifelse(tmp_mae$variable == "csr_mae",
                        "CSR Model", "Full Covariate Model")
tmp_mae$measure <- "MAE"

tmp_aic <- melt(results_200, id.vars = c("data_gen"),
            measure.vars = c("csr_aic", "AIC"))
tmp_aic$model <- ifelse(tmp_aic$variable == "csr_aic",
                        "CSR Model", "Full Covariate Model")
tmp_aic$measure <- "AIC"

tmp <- rbind(tmp_mae, tmp_aic)
tmp$model <- factor(tmp$model, levels = c("CSR Model", "Full Covariate Model"))
tmp$measure <- factor(tmp$measure, levels = c("MAE", "AIC"))

sim_labs <- c("CSR", "IP", "WIN", "pTHIN", "p(u)THIN", "P(u)THIN")
names(sim_labs) <- data_gen

model_pal <- c(cbPalette[2], cbPalette[4])

mae_ss <- ggplot(tmp, aes(y = data_gen, x = value, fill = model)) +
              geom_boxplot() +
              facet_wrap(~measure, ncol = 2, scales = "free_x",
                            strip.position = "bottom") +
              labs(y = "Point process") +
                theme(legend.position = "top", axis.title.x = element_blank(),
                        strip.background = element_blank(),
                        strip.placement = "outside") +
                scale_fill_manual(values = model_pal, name = "") +
              scale_y_discrete(labels = sim_labs, limits = rev)
ggsave(filename = "sim_200_mae_aic.pdf", plot = mae_ss,
        device = "pdf", path = path, width = 7, height = 3,
        units = "in", dpi = 600, limitsize = TRUE)

# plot variables
process_lab <- c("Sampling Window", "p-Thinning")
process <- c("WIN", "pTHIN")
names(process_lab) <- process
tmp <- variables[variables$dummy_prop == 10 &
                    variables$dummy_gen == "CSR" &
                    variables$data_gen %in% process &
                    variables$sample_size == 200, ]
data_gen_pal <- c(cbPalette[6], cbPalette[8])

var_plot <- ggplot(tmp, aes(x = term, y = wald, fill = data_gen)) +
                geom_boxplot() +
                annotate("rect", xmin = 0, xmax = 9, ymin = -2,
                         ymax = 2, alpha = 0.5, fill = "grey") +
                labs(x = "Covariate",
                  y = "Wald Statistic") +
                theme(legend.position = "top") +
                scale_fill_manual(name = "", labels = process_lab,
                  values = data_gen_pal) +
              scale_x_discrete(labels = sim_labs_x)
ggsave(filename = "sim_200_coef.pdf", plot = var_plot,
              device = "pdf", path = path, width = 7, height = 5,
              units = "in", dpi = 600, limitsize = TRUE)