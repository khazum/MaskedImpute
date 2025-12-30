#!/usr/bin/env Rscript
# install_imputation_libraries.R
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

cran_pkgs <- c("SAVER", "Matrix", "reticulate", "Rmagic")
bioc_pkgs <- c("SingleCellExperiment", "BiocParallel", "ccImpute")

install_cran(cran_pkgs)
install_bioc(bioc_pkgs)

cat("\nMAGIC Python dependency:\n")
cat("  pip install --user magic-impute\n")
cat("If Python or pip are missing, install Miniconda3 or another Python distribution.\n")
