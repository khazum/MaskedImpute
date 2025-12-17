#!/usr/bin/env Rscript
# run_imputation.R
# Usage: Rscript run_imputation.R <input_rds_file_or_dir> <output_dir> [ncores]
#
# Runs SAVER, ccImpute, and MAGIC on each input dataset and reports:
#   - overall MSE
#   - dropout MSE         (True>0 & Observed==0)
#   - biological zero MSE (True==0)
#   - non-dropout MSE     (True>0 & Observed>0)
#
# MSEs are computed in the same log2(1+normalized) space as experiment2.py,
# comparing imputed values to the dataset's `logTrueCounts`.
#
# Note: By default this script does NOT install missing R packages.
# Set `AUTO_INSTALL_PACKAGES=1` to enable BiocManager/install.packages calls.

stopf <- function(fmt, ...) stop(sprintf(fmt, ...), call. = FALSE)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stopf("Usage: Rscript run_imputation.R <input_rds_file_or_dir> <output_dir> [ncores]")
}

input_path <- args[1]
output_dir <- args[2]
ncores <- if (length(args) >= 3) as.integer(args[3]) else parallel::detectCores()
if (is.na(ncores) || ncores < 1) ncores <- 1

if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

options(repos = c(CRAN = "https://cran.rstudio.com/"))

auto_install <- tolower(Sys.getenv("AUTO_INSTALL_PACKAGES", unset = "0")) %in% c("1", "true", "yes")

ensure_pkg <- function(pkg, bioc = TRUE) {
  if (requireNamespace(pkg, quietly = TRUE)) return(invisible(TRUE))
  if (!auto_install) stopf("Missing required package '%s'. Install it (or set AUTO_INSTALL_PACKAGES=1).", pkg)
  if (!requireNamespace("BiocManager", quietly = TRUE)) {
    install.packages("BiocManager", quiet = FALSE)
  }
  if (bioc) {
    BiocManager::install(pkg, ask = FALSE, update = FALSE)
  } else {
    install.packages(pkg, quiet = FALSE)
  }
  if (!requireNamespace(pkg, quietly = TRUE)) stopf("Missing required package '%s'.", pkg)
  invisible(TRUE)
}

# Core R/Bioc packages (method-specific deps are checked per method)
required_pkgs <- c("SingleCellExperiment", "Matrix", "parallel")
invisible(lapply(required_pkgs, ensure_pkg))
invisible(lapply(required_pkgs, function(pkg) suppressPackageStartupMessages(library(pkg, character.only = TRUE))))

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
  log_imp <- as.matrix(log_imp)
  log_true <- as.matrix(log_true)
  log_obs <- as.matrix(log_obs)
  if (!all(dim(log_imp) == dim(log_true))) stopf("Dimension mismatch: log_imp vs log_true")
  if (!all(dim(log_obs) == dim(log_true))) stopf("Dimension mismatch: log_obs vs log_true")

  diff <- log_true - log_imp
  mask_biozero <- log_true == 0
  mask_dropout <- (log_true > 0) & (log_obs <= 0)
  mask_non_dropout <- (log_true > 0) & (log_obs > 0)

  mse_masked <- function(mask) {
    n <- sum(mask)
    if (n <= 0) return(NA_real_)
    mean((diff[mask])^2)
  }

  data.frame(
    mse = mean(diff^2),
    mse_dropout = mse_masked(mask_dropout),
    mse_biozero = mse_masked(mask_biozero),
    mse_non_dropout = mse_masked(mask_non_dropout),
    n_total = length(diff),
    n_dropout = sum(mask_dropout),
    n_biozero = sum(mask_biozero),
    n_non_dropout = sum(mask_non_dropout),
    stringsAsFactors = FALSE
  )
}

run_magic_logcounts <- function(logcounts_mat) {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stopf("MAGIC requires the R package 'reticulate' (install.packages('reticulate')).")
  }
  suppressPackageStartupMessages(library(reticulate))

  magic_python <- Sys.getenv("MAGIC_PYTHON", unset = "")
  if (nzchar(magic_python)) {
    reticulate::use_python(magic_python, required = TRUE)
  }

  if (!reticulate::py_available(initialize = FALSE)) {
    stopf("Python is not available to reticulate. Set MAGIC_PYTHON (or RETICULATE_PYTHON) to a Python executable with 'magic-impute' installed.")
  }

  magic <- reticulate::import("magic", delay_load = FALSE)
  if (is.null(magic$MAGIC)) {
    stopf("Python module 'magic' found but has no MAGIC class; install the KrishnaswamyLab MAGIC ('magic-impute').")
  }

  op <- tryCatch(
    magic$MAGIC(n_jobs = as.integer(ncores)),
    error = function(e) magic$MAGIC()
  )
  x_cells_genes <- t(as.matrix(logcounts_mat)) # cells x genes
  out <- op$fit_transform(x_cells_genes)
  out_r <- reticulate::py_to_r(out)
  out_mat <- as.matrix(out_r)

  expected <- c(ncol(logcounts_mat), nrow(logcounts_mat))
  if (!all(dim(out_mat) == expected)) {
    stopf(
      "Unexpected MAGIC output dimensions: got %s, expected %s.",
      paste(dim(out_mat), collapse = "x"),
      paste(expected, collapse = "x")
    )
  }

  out_genes_cells <- t(out_mat)
  rownames(out_genes_cells) <- rownames(logcounts_mat)
  colnames(out_genes_cells) <- colnames(logcounts_mat)
  out_genes_cells
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

methods <- c("saver", "ccimpute", "magic")
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

  mask_biozero <- log_true_f == 0
  mask_dropout <- (log_true_f > 0) & (log_obs_f <= 0)
  mask_non_dropout <- (log_true_f > 0) & (log_obs_f > 0)
  n_total <- length(log_true_f)
  n_dropout <- sum(mask_dropout)
  n_biozero <- sum(mask_biozero)
  n_non_dropout <- sum(mask_non_dropout)

  # --- SAVER ---
  saver_row <- tryCatch({
    cat("Running SAVER...\n")
    ensure_pkg("SAVER")
    saver_imp <- SAVER::saver(counts_f, ncores = ncores)
    save_imputed(dataset_name, saver_imp$estimate, "saver")
    log_imp <- normalize_counts_to_logcounts(saver_imp$estimate, denom_noisy, med_noisy)
    data.frame(
      dataset = dataset_name,
      method = "saver",
      compute_mse_metrics(log_imp, log_true_f, log_obs_f),
      error = NA_character_,
      stringsAsFactors = FALSE
    )
  }, error = function(e) {
    data.frame(
      dataset = dataset_name,
      method = "saver",
      mse = NA_real_,
      mse_dropout = NA_real_,
      mse_biozero = NA_real_,
      mse_non_dropout = NA_real_,
      n_total = n_total,
      n_dropout = n_dropout,
      n_biozero = n_biozero,
      n_non_dropout = n_non_dropout,
      error = conditionMessage(e),
      stringsAsFactors = FALSE
    )
  })
  all_results[[length(all_results) + 1]] <- saver_row

  # --- ccImpute ---
  cc_row <- tryCatch({
    cat("Running ccImpute...\n")
    ensure_pkg("ccImpute")

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

    cc_obj <- ccImpute::ccImpute(sce_f, k = n_groups, verbose = FALSE, BPPARAM = bpp)
    log_imp <- assay(cc_obj, "imputed")

    save_imputed(dataset_name, log_imp, "ccimpute")
    data.frame(
      dataset = dataset_name,
      method = "ccimpute",
      compute_mse_metrics(log_imp, log_true_f, log_obs_f),
      error = NA_character_,
      stringsAsFactors = FALSE
    )
  }, error = function(e) {
    data.frame(
      dataset = dataset_name,
      method = "ccimpute",
      mse = NA_real_,
      mse_dropout = NA_real_,
      mse_biozero = NA_real_,
      mse_non_dropout = NA_real_,
      n_total = n_total,
      n_dropout = n_dropout,
      n_biozero = n_biozero,
      n_non_dropout = n_non_dropout,
      error = conditionMessage(e),
      stringsAsFactors = FALSE
    )
  })
  all_results[[length(all_results) + 1]] <- cc_row

  # --- MAGIC ---
  magic_row <- tryCatch({
    cat("Running MAGIC...\n")
    magic_log <- run_magic_logcounts(log_obs_f)
    save_imputed(dataset_name, magic_log, "magic")
    data.frame(
      dataset = dataset_name,
      method = "magic",
      compute_mse_metrics(magic_log, log_true_f, log_obs_f),
      error = NA_character_,
      stringsAsFactors = FALSE
    )
  }, error = function(e) {
    data.frame(
      dataset = dataset_name,
      method = "magic",
      mse = NA_real_,
      mse_dropout = NA_real_,
      mse_biozero = NA_real_,
      mse_non_dropout = NA_real_,
      n_total = n_total,
      n_dropout = n_dropout,
      n_biozero = n_biozero,
      n_non_dropout = n_non_dropout,
      error = conditionMessage(e),
      stringsAsFactors = FALSE
    )
  })
  all_results[[length(all_results) + 1]] <- magic_row
}

if (length(all_results) == 0) stopf("No datasets processed.")

results_df <- do.call(rbind, all_results)

for (m in methods) {
  if (any(results_df$method == m)) write_method_table(results_df, m)
}

cat("\nDone.\n")
