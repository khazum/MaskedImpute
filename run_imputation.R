#!/usr/bin/env Rscript
# run_imputation.R
# Usage: Rscript run_imputation.R <input_rds_file_or_dir> <output_dir> [ncores] [n_repeats] [methods]
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
# Required packages: SingleCellExperiment, Matrix, parallel, SAVER, ccImpute, BiocParallel.
#
# Normalization follows the dataset's TrueCounts-derived library sizes and
# shared scale factor (median TrueCounts library size).

stopf <- function(fmt, ...) stop(sprintf(fmt, ...), call. = FALSE)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stopf("Usage: Rscript run_imputation.R <input_rds_file_or_dir> <output_dir> [ncores] [n_repeats] [methods]")
}

input_path <- args[1]
output_dir <- args[2]
ncores <- if (length(args) >= 3) as.integer(args[3]) else parallel::detectCores()
if (is.na(ncores) || ncores < 1) ncores <- 1
n_repeats <- if (length(args) >= 4) as.integer(args[4]) else 10L
if (is.na(n_repeats) || n_repeats < 1) n_repeats <- 1L
methods_arg <- if (length(args) >= 5) args[5] else "all"

if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

require_pkg <- function(pkg) {
  if (requireNamespace(pkg, quietly = TRUE)) return(invisible(TRUE))
  stopf("Missing required package '%s'. See header for dependencies.", pkg)
}

load_pkg <- function(pkg) {
  suppressPackageStartupMessages(library(pkg, character.only = TRUE))
}

report_error <- function(dataset, method, err_msg) {
  cat(sprintf("ERROR [%s/%s]: %s\n", dataset, method, err_msg))
}

# Core R/Bioc packages (method-specific deps are checked per method)
core_pkgs <- c("SingleCellExperiment", "Matrix", "parallel")
invisible(lapply(core_pkgs, require_pkg))
invisible(lapply(core_pkgs, load_pkg))

save_imputed <- function(dataset_name, data, method) {
  filename <- file.path(output_dir, paste0(dataset_name, "_", method, ".rds"))
  saveRDS(as.matrix(data), filename)
  cat(sprintf("  [%s] Saved imputed matrix to %s\n", method, filename))
}

# Extract TrueCounts-based library sizes, size factors, and shared scale factor.
get_normalization_info <- function(sce) {
  coldata <- colData(sce)
  md_norm <- metadata(sce)$normalization

  lib_true <- NULL
  if ("libSizeTrueCounts" %in% colnames(coldata)) {
    lib_true <- as.numeric(coldata$libSizeTrueCounts)
  } else if (!is.null(md_norm$library_sizes)) {
    lib_true <- as.numeric(md_norm$library_sizes)
  } else if ("TrueCounts" %in% assayNames(sce)) {
    lib_true <- colSums(assay(sce, "TrueCounts"))
  } else {
    stopf("Missing TrueCounts or libSizeTrueCounts for normalization.")
  }
  if (length(lib_true) != ncol(sce)) {
    stopf("libSizeTrueCounts length (%d) does not match number of cells (%d).",
          length(lib_true), ncol(sce))
  }

  scale_factor <- NA_real_
  if ("scaleFactorTrueCounts" %in% colnames(coldata)) {
    scale_vals <- unique(as.numeric(coldata$scaleFactorTrueCounts))
    scale_vals <- scale_vals[is.finite(scale_vals) & scale_vals > 0]
    if (length(scale_vals) > 0) scale_factor <- scale_vals[1]
  }
  if (!is.finite(scale_factor) || scale_factor <= 0) {
    if (!is.null(md_norm$scale_factor)) {
      scale_factor <- as.numeric(md_norm$scale_factor)[1]
    }
  }
  if (!is.finite(scale_factor) || scale_factor <= 0) {
    scale_factor <- stats::median(lib_true[lib_true > 0])
  }
  if (!is.finite(scale_factor) || scale_factor <= 0) scale_factor <- 1

  size_factor <- NULL
  if ("sizeFactorTrueCounts" %in% colnames(coldata)) {
    size_factor <- as.numeric(coldata$sizeFactorTrueCounts)
  } else if ("sizeFactorsTrueCounts" %in% colnames(coldata)) {
    size_factor <- as.numeric(coldata$sizeFactorsTrueCounts)
  } else if (!is.null(md_norm$size_factors)) {
    size_factor <- as.numeric(md_norm$size_factors)
  }
  if (is.null(size_factor)) {
    size_factor <- lib_true / scale_factor
  }
  if (length(size_factor) != ncol(sce)) {
    stopf("sizeFactorTrueCounts length (%d) does not match number of cells (%d).",
          length(size_factor), ncol(sce))
  }
  size_factor <- as.numeric(size_factor)

  list(lib_true = lib_true, scale_factor = scale_factor, size_factor = size_factor)
}

# Convert counts to log2(1+normalized) using TrueCounts size factors.
normalize_counts_to_logcounts <- function(counts, size_factor) {
  counts <- as.matrix(counts)
  if (ncol(counts) != length(size_factor)) {
    stopf("Size factor length (%d) does not match number of cells (%d).",
          length(size_factor), ncol(counts))
  }
  denom <- ifelse(is.finite(size_factor) & size_factor > 0, size_factor, 1)
  log2(1 + t(t(counts) / denom))
}

# Precompute masks for dropout and biological zero stratification.
compute_masks <- function(log_true, log_obs) {
  epsilon <- 1e-6
  list(
    biozero = log_true <= epsilon,
    dropout = (log_true > epsilon) & (log_obs <= epsilon),
    non_zero = (log_true > epsilon) & (log_obs > epsilon)
  )
}

compute_mse_metrics <- function(log_imp, log_true, masks) {
  log_imp <- as.matrix(log_imp)
  log_true <- as.matrix(log_true)
  if (!all(dim(log_imp) == dim(log_true))) stopf("Dimension mismatch: log_imp vs log_true")

  diff <- log_true - log_imp

  mse_masked <- function(mask) {
    n <- sum(mask)
    if (n <= 0) return(NA_real_)
    mean((diff[mask])^2)
  }

  data.frame(
    mse = mean(diff^2),
    mse_dropout = mse_masked(masks$dropout),
    mse_biozero = mse_masked(masks$biozero),
    mse_non_zero = mse_masked(masks$non_zero),
    n_total = length(diff),
    n_dropout = sum(masks$dropout),
    n_biozero = sum(masks$biozero),
    n_non_zero = sum(masks$non_zero),
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

methods <- parse_methods(methods_arg)
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
  log_obs <- as.matrix(assay(sce, "logcounts"))
  log_true <- as.matrix(assay(sce, "logTrueCounts"))

  if (!all(dim(counts) == dim(log_obs)) || !all(dim(log_true) == dim(log_obs))) {
    stopf("[%s] Assay dimension mismatch among counts/logcounts/logTrueCounts.", dataset_name)
  }

  norm_info <- get_normalization_info(sce)
  size_factor_true <- as.numeric(norm_info$size_factor)

  masks <- compute_masks(log_true, log_obs)
  n_total <- length(log_true)
  n_dropout <- sum(masks$dropout)
  n_biozero <- sum(masks$biozero)
  n_non_zero <- sum(masks$non_zero)

  if ("baseline" %in% methods) {
    # --- Baseline (no imputation) ---
    cat(sprintf("Running baseline (no imputation) x%d...\n", n_repeats))
    baseline_metrics <- list()
    baseline_runtimes <- numeric()
    baseline_err <- NA_character_
    for (i in seq_len(n_repeats)) {
      t0 <- proc.time()
      res <- tryCatch({
        log_imp <- log_obs
        compute_mse_metrics(log_imp, log_true, masks)
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
  }

  if ("saver" %in% methods) {
    # --- SAVER ---
    cat(sprintf("Running SAVER x%d...\n", n_repeats))
    require_pkg("SAVER")
    # SAVER drops zero-count cells internally; pre-filter and align normalization.
    lib_obs <- Matrix::colSums(counts)
    nonzero_cells <- lib_obs > 0
    if (!all(nonzero_cells)) {
      cat(sprintf("  [saver] Dropping %d zero-expression cells.\n", sum(!nonzero_cells)))
    }
    valid_sf <- is.finite(size_factor_true) & size_factor_true > 0
    if (!all(valid_sf)) {
      cat(sprintf("  [saver] Dropping %d cells with invalid TrueCounts size factors.\n", sum(!valid_sf)))
    }
    saver_cells <- nonzero_cells & valid_sf
    counts_saver <- Matrix::Matrix(counts[, saver_cells, drop = FALSE], sparse = TRUE)
    if (ncol(counts_saver) < 2 || nrow(counts_saver) < 1) {
      saver_err <- "Insufficient cells or genes after filtering."
      report_error(dataset_name, "saver", saver_err)
      saver_metrics <- list()
      saver_runtimes <- numeric()
    } else {
      mean_obs <- mean(lib_obs[saver_cells])
      if (!is.finite(mean_obs) || mean_obs <= 0) mean_obs <- 1
      size_factor_obs <- lib_obs[saver_cells] / mean_obs
      # If no predictor genes pass SAVER's mean threshold, fall back to null model.
      use_null_model <- all(Matrix::rowMeans(counts_saver) < 0.1)
      if (use_null_model) {
        cat("  [saver] Using null model (no predictor genes above mean threshold).\n")
      }

      saver_ncores <- ncores
      if (saver_ncores > 1) {
        message("SAVER can be unstable with ncores > 1; using ncores = 1.")
        saver_ncores <- 1
      }
      saver_metrics <- list()
      saver_runtimes <- numeric()
      saver_err <- NA_character_
      for (i in seq_len(n_repeats)) {
        attempt_null_model <- use_null_model
        repeat {
          t0 <- proc.time()
          res <- tryCatch({
            saver_imp <- SAVER::saver(
              counts_saver,
              ncores = saver_ncores,
              null.model = attempt_null_model
            )
            saver_est <- as.matrix(saver_imp$estimate)
            saver_counts <- sweep(saver_est, 2, size_factor_obs, "*")
            saver_norm <- sweep(saver_counts, 2, size_factor_true[saver_cells], "/")
            saver_est_full <- matrix(0, nrow = nrow(counts), ncol = ncol(counts), dimnames = dimnames(counts))
            saver_est_full[, saver_cells] <- saver_norm
            if (i == 1) save_imputed(dataset_name, saver_est_full, "saver")

            log_imp <- log_obs
            log_imp[, saver_cells] <- normalize_counts_to_logcounts(
              saver_counts,
              size_factor_true[saver_cells]
            )
            compute_mse_metrics(log_imp, log_true, masks)
          }, error = function(e) {
            saver_err <<- conditionMessage(e)
            report_error(dataset_name, "saver", saver_err)
            NULL
          })
          elapsed <- (proc.time() - t0)["elapsed"]
          if (is.null(res) && !attempt_null_model) {
            cat("  [saver] Retrying with null model due to error.\n")
            attempt_null_model <- TRUE
            next
          }
          break
        }
        if (is.null(res)) break
        use_null_model <- attempt_null_model
        saver_metrics[[length(saver_metrics) + 1]] <- res
        saver_runtimes <- c(saver_runtimes, elapsed)
      }
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
  }

  if ("ccimpute" %in% methods) {
    # --- ccImpute ---
    cat(sprintf("Running ccImpute x%d...\n", n_repeats))
    require_pkg("ccImpute")
    require_pkg("BiocParallel")

    n_groups <- NA_integer_
    if (!is.null(colData(sce)$Group)) n_groups <- length(unique(colData(sce)$Group))
    if (!is.finite(n_groups) || n_groups < 2) n_groups <- 2

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
        cc_obj <- ccImpute::ccImpute(sce, k = n_groups, verbose = FALSE, BPPARAM = bpp)
        log_imp <- assay(cc_obj, "imputed")
        if (i == 1) save_imputed(dataset_name, log_imp, "ccimpute")
        compute_mse_metrics(log_imp, log_true, masks)
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

}

if (length(all_results) == 0) stopf("No datasets processed.")

results_df <- do.call(rbind, all_results)

for (m in methods) {
  if (any(results_df$method == m)) write_method_table(results_df, m)
}

cat("\nDone.\n")
parse_methods <- function(raw) {
  if (is.null(raw) || !nzchar(raw) || tolower(raw) == "all") {
    return(c("baseline", "saver", "ccimpute"))
  }
  methods <- tolower(unlist(strsplit(raw, ",")))
  methods <- methods[nzchar(methods)]
  allowed <- c("baseline", "saver", "ccimpute")
  unknown <- setdiff(methods, allowed)
  if (length(unknown) > 0) {
    stopf("Unknown methods: %s. Allowed: %s or 'all'.",
          paste(unknown, collapse = ", "), paste(allowed, collapse = ", "))
  }
  unique(methods)
}
