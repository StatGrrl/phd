library(spatstat)
library(raster)
library(sf)
library(maptools)
library(rgdal)

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
covariates <- list(dist_roads, dist_camps, dist_picnic, dist_gates,
            dist_border, dist_dams, dist_water)
names <- c("roads", "camps", "picnic", "gates", "border", "dams", "water")
m_to_km <- function(x) x / 1000
covariates <- lapply(covariates, m_to_km)
covariates <- lapply(covariates, as.im.RasterLayer)
covariates <- setNames(covariates, names)
covariates <- lapply(covariates, rescale, s = 1000, unitname = "km")
all_images <- as.solist(c(list(poach = poach_prob), covariates))
covariates <- as.solist(covariates)

# plot pixel images
xlab <- "Easting distance (km)"
ylab <- "Northing distance (km)"
panel <- c("(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)")
pdf(paste0(path, "cov_images.pdf"), height = 9, width = 7)
par(mfrow = c(4, 2), mar = c(2, 1, 1, 3), cex.main = 0.8, oma = c(3, 3, 0, 0))
for (i in seq_along(all_images)) {
  rlab <- ifelse(i == 1, "Probability", "Distance (km)")
  plot(reflect(all_images[[i]]), axes = TRUE, ribsep = 0.05, main = "")
  mtext(rlab, side = 4, line = 1.75, cex = 0.8)
  #mtext(bquote(italic(.(panel[i]))), side = 3, line = -0.5, cex = 0.8)
  if (i == 1) {
    mtext(xlab, side = 1, line = 0.5, cex = 0.8, outer = TRUE)
    mtext(ylab, side = 2, line = 1, cex = 0.8, outer = TRUE)
  }
}
dev.off()

# remove unneeded objects and save
rm(list = c("s", "r", "dist_roads", "dist_camps", "dist_picnic",
            "dist_gates", "dist_border", "dist_dams", "dist_water",
            "all_images", "xlab", "ylab", "panel", "i", "rlab",
            "m_to_km", "names"))
save.image(file = paste0(path, "knp_poach.RData"))