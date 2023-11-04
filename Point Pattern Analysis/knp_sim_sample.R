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
results <- hyperframe(iter = 0, sim_data = "CSR", sample_size = 50,
                        pp = ppp(0, 0)) # dummy record, delete later
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
        hf_iter <- hyperframe(iter = i,
                                sim_data = sim_data[j],
                                sample_size = sample_size[n],
                                pp = pp)
        results <- rbind.hyperframe(results, hf_iter)
    }
  }
}

# remove simulation variables
rm(iter, i, j, n, p, p_u, n_large, lambda, lambda_u, pp, pp_large, hf_iter)
results <- results[-1, ] # delete dummy record

# factor variables
results$sim_data <- factor(results$sim_data, levels = sim_data)
results$sample_size <- factor(results$sample_size, levels = sample_size)

# data frame for plotting
results_df <- as.data.frame(results)

# descriptive statistics
results_df$npoints <- with(results, npoints(pp))
results_df$intensity <- with(results, intensity(pp))

# fit models
results_fit <- results[, 1:3]
results_fit$fit_csr <- with(results, ppm(pp, ~1))
results_fit$fit_cov <- with(results, ppm(pp, ~ ., covariates = covariates))

# quadrat gof test
results_fit$gof_csr <- with(results_fit, quadrat.test(fit_csr, nx = 4, ny = 3))
results_fit$gof_cov <- with(results_fit, quadrat.test(fit_cov, nx = 4, ny = 3))

# mean absolute error
results_fit$npoints <- with(results, npoints(pp))
results_df$mae_csr <- with(results_fit,
            sum(abs(residuals(fit_csr)$val)) / length(residuals(fit_csr)$val))
results_df$mae_cov <- with(results_fit,
            sum(abs(residuals(fit_cov)$val)) / length(residuals(fit_cov)$val))

# goodness of fit
results_df$gof_csr <- with(results_fit, gof_csr$p.value)
results_df$gof_cov <- with(results_fit, gof_cov$p.value)

# plots of mean absolute error
s_mae_box <- ggplot(results_df, aes(y = sample_size, x = mae_cov)) +
              geom_boxplot() +
              facet_wrap(~sim_data, ncol = 2) +
              labs(y = "Sample size") +
              theme(axis.title.x = element_blank())
s_mae_dens <- ggplot(results_df, aes(mae_cov, 
              color = sample_size)) +
                geom_density(alpha = 0.3) +
                facet_wrap(~sim_data, ncol = 2) +
                labs(x = "Mean absolute error", y = "Density") +
                scale_color_discrete(name = "Sample size",
                  guide = guide_legend(nrow = 2)) +
                theme(legend.position = c(0.83, 0.9),
                  legend.title = element_blank(),
                  legend.text = element_text(size = 10),
                  legend.key.size = unit(0.2, "cm"),
                  legend.background = element_blank(),
                  legend.box.background = element_rect(color = "black"))
s_mae <- s_mae_box / s_mae_dens
ggsave(filename = "sim_sample_mae.pdf", plot = s_mae,
        device = "pdf", path = path, width = 7, height = 9,
        units = "in", dpi = 500, limitsize = TRUE)

# plot gof rejection rate
results_df["gof_cov_H0"] <- ifelse(results_df$gof_cov < 0.05,
                                      "Reject", "Do not reject")
results_df["gof_csr_H0"] <- ifelse(results_df$gof_csr < 0.05,
                                      "Reject", "Do not reject")
results_df$gof_cov_H0 <- factor(results_df$gof_cov_H0,
                                  levels = c("Reject", "Do not reject"))
results_df$gof_csr_H0 <- factor(results_df$gof_csr_H0,
                                  levels = c("Reject", "Do not reject"))
tmp <- melt(results_df, id.vars = c("sim_data", "sample_size"),
            measure.vars = c("gof_csr_H0", "gof_cov_H0"))
tmp <- tmp[tmp$value == "Reject", ]
gof_labs <- c("Reject CSR model", "Reject covariate model")
names(gof_labs) <- c("gof_csr_H0", "gof_cov_H0")
s_gof <- ggplot(tmp, aes(x = sample_size,
                      fill = variable)) +
              geom_bar(aes(y = (..count..) / 1000),
                  position =   position_dodge()) +
              facet_wrap(~sim_data, ncol = 3) +
              labs(x = "Sample size", y = "Percentage of point patterns") +
              theme(legend.position = "top") +
          scale_y_continuous(labels = scales::percent) +
              scale_fill_discrete(name = "", labels = gof_labs,
                guide = guide_legend(nrow = 1))
ggsave(filename = "sim_sample_gof.pdf", plot = s_gof,
        device = "pdf", path = path, width = 8, height = 4,
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
