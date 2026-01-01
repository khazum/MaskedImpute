#!/usr/bin/env Rscript
# install_imputation_libs.R
# Installs R dependencies for run_imputation.R.

options(repos = c(CRAN = "https://cran.rstudio.com/"))

install_cran <- function(pkgs) {
  for (pkg in pkgs) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
      install.packages(pkg, quiet = FALSE)
    }
  }
}

install_bioc <- function(pkgs) {
  if (!requireNamespace("BiocManager", quietly = TRUE)) {
    install.packages("BiocManager", quiet = FALSE)
  }
  for (pkg in pkgs) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
      BiocManager::install(pkg, ask = FALSE, update = FALSE)
    }
  }
}

cran_pkgs <- c("SAVER", "Matrix")
bioc_pkgs <- c("SingleCellExperiment", "BiocParallel", "ccImpute")

install_cran(cran_pkgs)
install_bioc(bioc_pkgs)
