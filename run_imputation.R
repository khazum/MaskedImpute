#!/usr/bin/env Rscript
# run_imputation.R
# Usage: Rscript run_imputation.R <input_rds_file_or_dir> <output_dir> [ncores] [n_repeats]
#
# Runs SAVER and ccImpute on each input dataset and reports:
#   - overall MSE
#   - dropout MSE         (True>0 & Observed==0)
#   - biological zero MSE (True==0)
#   - non-zero MSE        (True>0 & Observed>0)
#
# MSEs are computed in the same log2(1+normalized) space as experiment2.py,
# comparing imputed values to the dataset's `logTrueCounts`.
#
# Note: This script does NOT install missing R packages.
# See install_imputation_libs.R for dependency installation.

stopf <- function(fmt, ...) stop(sprintf(fmt, ...), call. = FALSE)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stopf("Usage: Rscript run_imputation.R <input_rds_file_or_dir> <output_dir> [ncores] [n_repeats]")
}

input_path <- args[1]
output_dir <- args[2]
ncores <- if (length(args) >= 3) as.integer(args[3]) else parallel::detectCores()
if (is.na(ncores) || ncores < 1) ncores <- 1
n_repeats <- if (length(args) >= 4) as.integer(args[4]) else 10L
if (is.na(n_repeats) || n_repeats < 1) n_repeats <- 1L

if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

options(repos = c(CRAN = "https://cran.rstudio.com/"))

bioc_install_hint <- function(pkg) {
  sprintf(
    "Install via Bioconductor:\nif (!require(\"BiocManager\", quietly = TRUE)) install.packages(\"BiocManager\")\nBiocManager::install(\"%s\")",
    pkg
  )
}

cran_install_hint <- function(pkg) {
  sprintf("Install from CRAN:\ninstall.packages(\"%s\")", pkg)
}

require_pkg <- function(pkg, source = "cran") {
  if (requireNamespace(pkg, quietly = TRUE)) return(invisible(TRUE))
  hint <- switch(
    source,
    bioc = bioc_install_hint(pkg),
    cran = cran_install_hint(pkg),
    base = "This package should be available with base R.",
    ""
  )
  stopf("Missing required package '%s'. %s", pkg, hint)
}

load_pkg <- function(pkg) {
  suppressPackageStartupMessages(library(pkg, character.only = TRUE))
}

report_error <- function(dataset, method, err_msg) {
  cat(sprintf("ERROR [%s/%s]: %s\n", dataset, method, err_msg))
}

# Core R/Bioc packages (method-specific deps are checked per method)
core_pkgs <- list(
  list(name = "SingleCellExperiment", source = "bioc"),
  list(name = "Matrix", source = "cran"),
  list(name = "parallel", source = "base")
)
invisible(lapply(core_pkgs, function(p) require_pkg(p$name, p$source)))
invisible(lapply(core_pkgs, function(p) load_pkg(p$name)))

save_imputed <- function(dataset_name, data, method) {
  filename <- file.path(output_dir, paste0(dataset_name, "_", method, ".rds"))
  saveRDS(as.matrix(data), filename)
  cat(sprintf("  [%s] Saved imputed matrix to %s\n", method, filename))
}

normalize_counts_to_logcounts <- function(counts_mat, denom_noisy, med_noisy) {
  x <- as.matrix(counts_mat)
  x <- pmax(x, 0)
  norm <- t(t(x) / denom_noisy * med_noisy)
  log2(1 + norm)
}

compute_mse_metrics <- function(log_imp, log_true, log_obs) {
  epsilon <- 1e-6
  log_imp <- as.matrix(log_imp)
  log_true <- as.matrix(log_true)
  log_obs <- as.matrix(log_obs)
  if (!all(dim(log_imp) == dim(log_true))) stopf("Dimension mismatch: log_imp vs log_true")
  if (!all(dim(log_obs) == dim(log_true))) stopf("Dimension mismatch: log_obs vs log_true")

  diff <- log_true - log_imp
  mask_biozero <- log_true <= epsilon
  mask_dropout <- (log_true > epsilon) & (log_obs <= epsilon)
  mask_non_zero <- (log_true > epsilon) & (log_obs > epsilon)

  mse_masked <- function(mask) {
    n <- sum(mask)
    if (n <= 0) return(NA_real_)
    mean((diff[mask])^2)
  }

  data.frame(
    mse = mean(diff^2),
    mse_dropout = mse_masked(mask_dropout),
    mse_biozero = mse_masked(mask_biozero),
    mse_non_zero = mse_masked(mask_non_zero),
    n_total = length(diff),
    n_dropout = sum(mask_dropout),
    n_biozero = sum(mask_biozero),
    n_non_zero = sum(mask_non_zero),
    stringsAsFactors = FALSE
  )
}

summarize_repeats <- function(metrics_list, runtimes, n_total, n_dropout, n_biozero, n_non_zero, err_msg) {
  metric_cols <- c("mse", "mse_dropout", "mse_biozero", "mse_non_zero")
  if (length(metrics_list) > 0) {
    metrics_df <- do.call(rbind, metrics_list)
    means <- sapply(metric_cols, function(cn) mean(metrics_df[[cn]], na.rm = TRUE))
    sds <- sapply(metric_cols, function(cn) if (nrow(metrics_df) > 1) stats::sd(metrics_df[[cn]], na.rm = TRUE) else 0)
    runtime_mean <- mean(runtimes)
    runtime_sd <- if (length(runtimes) > 1) stats::sd(runtimes) else 0
    data.frame(
      mse = means["mse"],
      mse_std = sds["mse"],
      mse_dropout = means["mse_dropout"],
      mse_dropout_std = sds["mse_dropout"],
      mse_biozero = means["mse_biozero"],
      mse_biozero_std = sds["mse_biozero"],
      mse_non_zero = means["mse_non_zero"],
      mse_non_zero_std = sds["mse_non_zero"],
      runtime_sec = runtime_mean,
      runtime_sec_std = runtime_sd,
      n_repeats = length(runtimes),
      n_total = n_total,
      n_dropout = n_dropout,
      n_biozero = n_biozero,
      n_non_zero = n_non_zero,
      error = err_msg,
      stringsAsFactors = FALSE
    )
  } else {
    data.frame(
      mse = NA_real_,
      mse_std = NA_real_,
      mse_dropout = NA_real_,
      mse_dropout_std = NA_real_,
      mse_biozero = NA_real_,
      mse_biozero_std = NA_real_,
      mse_non_zero = NA_real_,
      mse_non_zero_std = NA_real_,
      runtime_sec = NA_real_,
      runtime_sec_std = NA_real_,
      n_repeats = 0L,
      n_total = n_total,
      n_dropout = n_dropout,
      n_biozero = n_biozero,
      n_non_zero = n_non_zero,
      error = err_msg,
      stringsAsFactors = FALSE
    )
  }
}

write_method_table <- function(results_df, method) {
  out_path <- file.path(output_dir, paste0(method, "_mse_table.tsv"))
  out_df <- results_df[results_df$method == method, , drop = FALSE]
  out_df$method <- NULL

  if (file.exists(out_path)) {
    existing <- tryCatch(
      read.delim(out_path, sep = "\t", stringsAsFactors = FALSE, check.names = FALSE),
      error = function(e) NULL
    )
    if (!is.null(existing) && ("dataset" %in% names(existing))) {
      if (!identical(names(existing), names(out_df))) {
        warning(sprintf("Existing table has different columns; overwriting: %s", out_path))
      } else {
        existing <- existing[!(existing$dataset %in% out_df$dataset), , drop = FALSE]
        out_df <- rbind(existing, out_df)
      }
    }
  }

  out_df <- out_df[order(out_df$dataset), , drop = FALSE]
  write.table(out_df, out_path, sep = "\t", quote = FALSE, row.names = FALSE)
  cat(sprintf("Wrote %s\n", out_path))
}

# Expand input path to files
input_files <- character()
if (dir.exists(input_path)) {
  input_files <- list.files(input_path, pattern = "\\.rds$", full.names = TRUE, recursive = TRUE)
} else if (file.exists(input_path)) {
  input_files <- input_path
} else {
  stopf("Input path not found: %s", input_path)
}
if (length(input_files) == 0) stopf("No .rds files found under: %s", input_path)

methods <- c("baseline", "saver", "ccimpute")
all_results <- list()

for (input_file in input_files) {
  dataset_name <- tools::file_path_sans_ext(basename(input_file))
  cat(sprintf("\n=== %s ===\n", dataset_name))

  sce <- readRDS(input_file)
  needed_assays <- c("counts", "logcounts", "logTrueCounts")
  missing_assays <- setdiff(needed_assays, assayNames(sce))
  if (length(missing_assays) > 0) {
    warning(sprintf("[%s] Missing assays: %s (skipping).", dataset_name, paste(missing_assays, collapse = ", ")))
    next
  }

  counts <- assay(sce, "counts") # genes x cells
  log_obs <- assay(sce, "logcounts")
  log_true <- assay(sce, "logTrueCounts")

  keep_genes <- Matrix::rowSums(counts > 0) >= 2
  counts_f <- counts[keep_genes, , drop = FALSE]
  log_obs_f <- log_obs[keep_genes, , drop = FALSE]
  log_true_f <- log_true[keep_genes, , drop = FALSE]
  cat(sprintf("Genes: %d -> %d (filtered)\n", nrow(counts), nrow(counts_f)))

  if (nrow(counts_f) == 0 || ncol(counts_f) == 0) {
    warning(sprintf("[%s] Empty filtered matrix (skipping).", dataset_name))
    next
  }

  lib_noisy <- colSums(counts)
  denom_noisy <- ifelse(lib_noisy > 0, lib_noisy, 1)
  med_noisy <- stats::median(lib_noisy[lib_noisy > 0])
  if (!is.finite(med_noisy) || med_noisy <= 0) med_noisy <- 1

  epsilon <- 1e-6
  mask_biozero <- log_true_f <= epsilon
  mask_dropout <- (log_true_f > epsilon) & (log_obs_f <= epsilon)
  mask_non_zero <- (log_true_f > epsilon) & (log_obs_f > epsilon)
  n_total <- length(log_true_f)
  n_dropout <- sum(mask_dropout)
  n_biozero <- sum(mask_biozero)
  n_non_zero <- sum(mask_non_zero)

  # --- Baseline (no imputation) ---
  cat(sprintf("Running baseline (no imputation) x%d...\n", n_repeats))
  baseline_metrics <- list()
  baseline_runtimes <- numeric()
  baseline_err <- NA_character_
  for (i in seq_len(n_repeats)) {
    t0 <- proc.time()
    res <- tryCatch({
      log_imp <- log_obs_f
      compute_mse_metrics(log_imp, log_true_f, log_obs_f)
    }, error = function(e) {
      baseline_err <<- conditionMessage(e)
      report_error(dataset_name, "baseline", baseline_err)
      NULL
    })
    elapsed <- (proc.time() - t0)["elapsed"]
    if (is.null(res)) break
    baseline_metrics[[length(baseline_metrics) + 1]] <- res
    baseline_runtimes <- c(baseline_runtimes, elapsed)
  }
  baseline_row <- data.frame(
    dataset = dataset_name,
    method = "baseline",
    summarize_repeats(
      baseline_metrics,
      baseline_runtimes,
      n_total,
      n_dropout,
      n_biozero,
      n_non_zero,
      baseline_err
    ),
    stringsAsFactors = FALSE
  )
  all_results[[length(all_results) + 1]] <- baseline_row

  # --- SAVER ---
  cat(sprintf("Running SAVER x%d...\n", n_repeats))
  require_pkg("SAVER", "cran")
  counts_saver <- Matrix::Matrix(counts_f, sparse = TRUE)
  saver_ncores <- ncores
  if (saver_ncores > 1) {
    message("SAVER can be unstable with ncores > 1; using ncores = 1.")
    saver_ncores <- 1
  }
  saver_metrics <- list()
  saver_runtimes <- numeric()
  saver_err <- NA_character_
  for (i in seq_len(n_repeats)) {
    t0 <- proc.time()
    res <- tryCatch({
      saver_imp <- SAVER::saver(counts_saver, ncores = saver_ncores)
      if (i == 1) save_imputed(dataset_name, saver_imp$estimate, "saver")
      log_imp <- normalize_counts_to_logcounts(saver_imp$estimate, denom_noisy, med_noisy)
      compute_mse_metrics(log_imp, log_true_f, log_obs_f)
    }, error = function(e) {
      saver_err <<- conditionMessage(e)
      report_error(dataset_name, "saver", saver_err)
      NULL
    })
    elapsed <- (proc.time() - t0)["elapsed"]
    if (is.null(res)) break
    saver_metrics[[length(saver_metrics) + 1]] <- res
    saver_runtimes <- c(saver_runtimes, elapsed)
  }
  saver_row <- data.frame(
    dataset = dataset_name,
    method = "saver",
    summarize_repeats(
      saver_metrics,
      saver_runtimes,
      n_total,
      n_dropout,
      n_biozero,
      n_non_zero,
      saver_err
    ),
    stringsAsFactors = FALSE
  )
  all_results[[length(all_results) + 1]] <- saver_row

  # --- ccImpute ---
  cat(sprintf("Running ccImpute x%d...\n", n_repeats))
  require_pkg("ccImpute", "bioc")
  require_pkg("BiocParallel", "bioc")

  n_groups <- NA_integer_
  if (!is.null(colData(sce)$Group)) n_groups <- length(unique(colData(sce)$Group))
  if (!is.finite(n_groups) || n_groups < 2) n_groups <- 2

  sce_f <- sce[keep_genes, , drop = FALSE]
  bpp <- BiocParallel::SerialParam()
  if (ncores > 1) {
    bpp <- tryCatch(
      BiocParallel::MulticoreParam(workers = ncores),
      error = function(e) BiocParallel::SerialParam()
    )
  }

  cc_metrics <- list()
  cc_runtimes <- numeric()
  cc_err <- NA_character_
  for (i in seq_len(n_repeats)) {
    t0 <- proc.time()
    res <- tryCatch({
      cc_obj <- ccImpute::ccImpute(sce_f, k = n_groups, verbose = FALSE, BPPARAM = bpp)
      log_imp <- assay(cc_obj, "imputed")
      if (i == 1) save_imputed(dataset_name, log_imp, "ccimpute")
      compute_mse_metrics(log_imp, log_true_f, log_obs_f)
    }, error = function(e) {
      cc_err <<- conditionMessage(e)
      report_error(dataset_name, "ccimpute", cc_err)
      NULL
    })
    elapsed <- (proc.time() - t0)["elapsed"]
    if (is.null(res)) break
    cc_metrics[[length(cc_metrics) + 1]] <- res
    cc_runtimes <- c(cc_runtimes, elapsed)
  }
  cc_row <- data.frame(
    dataset = dataset_name,
    method = "ccimpute",
    summarize_repeats(
      cc_metrics,
      cc_runtimes,
      n_total,
      n_dropout,
      n_biozero,
      n_non_zero,
      cc_err
    ),
    stringsAsFactors = FALSE
  )
  all_results[[length(all_results) + 1]] <- cc_row

}

if (length(all_results) == 0) stopf("No datasets processed.")

results_df <- do.call(rbind, all_results)

for (m in methods) {
  if (any(results_df$method == m)) write_method_table(results_df, m)
}

cat("\nDone.\n")
