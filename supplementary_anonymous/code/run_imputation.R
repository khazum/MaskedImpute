#!/usr/bin/env Rscript
# run_imputation.R
# Usage: Rscript run_imputation.R <input_rds_file_or_dir> <output_dir> [ncores] [n_repeats] [methods]
#
# Runs baseline/SAVER/ccImpute on each input dataset and reports benchmark
# reconstruction metrics in log2(1+normalized) space vs. `logTrueCounts`:
#   - MSE / MAE (overall, dropout, biological zero, non-zero)
#   - MSE / MAE on marker genes (DEFacGroup columns in rowData)
#   - per-gene normalized RMSE (gNRMSE)
#   - gNRMSE on marker genes (DEFacGroup columns in rowData)
#   - gene-gene correlation error (CorrErr)
#
# Required packages: SingleCellExperiment, Matrix, parallel, SAVER, ccImpute, BiocParallel.
#
# Count outputs are normalized with dataset `target_sum` and library sizes
# recomputed from each imputed count matrix (no libSizeTrue usage).

stopf <- function(fmt, ...) stop(sprintf(fmt, ...), call. = FALSE)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stopf("Usage: Rscript run_imputation.R <input_rds_file_or_dir> <output_dir> [ncores] [n_repeats] [methods]")
}

input_path <- args[1]
output_dir <- args[2]
ncores <- if (length(args) >= 3) as.integer(args[3]) else parallel::detectCores()
if (is.na(ncores) || ncores < 1) ncores <- 1
saver_default_cores <- suppressWarnings(as.integer(Sys.getenv("SAVER_CORES", "8")))
if (is.na(saver_default_cores) || saver_default_cores < 1) saver_default_cores <- 8
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

save_imputed <- function(dataset_name, data, method) {
  filename <- file.path(output_dir, paste0(dataset_name, "_", method, ".rds"))
  saveRDS(as.matrix(data), filename)
  cat(sprintf("  [%s] Saved imputed matrix to %s\n", method, filename))
}

# Extract target library-size scaling constant for CP normalization.
get_normalization_info <- function(sce) {
  coldata <- colData(sce)
  md_norm <- metadata(sce)$normalization

  target_sum <- NA_real_
  if ("targetSum" %in% colnames(coldata)) {
    vals <- as.numeric(coldata$targetSum)
    vals <- vals[is.finite(vals) & vals > 0]
    if (length(vals) > 0) target_sum <- vals[1]
  }
  if (!is.null(md_norm$target_sum)) {
    if (!is.finite(target_sum) || target_sum <= 0) {
      target_sum <- as.numeric(md_norm$target_sum)[1]
    }
  }
  if (!is.finite(target_sum) || target_sum <= 0) {
    stopf("Missing/invalid normalization$target_sum metadata (libSizeTrue fallback is disabled).")
  }
  list(target_sum = target_sum)
}

# Convert counts to log2(1+CPtarget) using library sizes computed from the given counts.
normalize_counts_to_logcounts <- function(counts, target_sum) {
  counts <- as.matrix(counts)
  target_sum <- as.numeric(target_sum)[1]
  if (!is.finite(target_sum) || target_sum <= 0) {
    stopf("Invalid target_sum for normalization: %s", as.character(target_sum))
  }
  lib_sizes <- colSums(counts)
  denom <- ifelse(is.finite(lib_sizes) & lib_sizes > 0, lib_sizes / target_sum, 1)
  log2(1 + t(t(counts) / denom))
}

extract_marker_gene_mask <- function(sce, epsilon = 1e-8) {
  rd <- SummarizedExperiment::rowData(sce)
  defac_cols <- grep("^DEFac", colnames(rd), value = TRUE)
  if (length(defac_cols) == 0) {
    warning("Missing DEFac rowData columns; marker-subset metrics will be NA.")
    return(rep(FALSE, nrow(sce)))
  }
  marker_gene <- rep(FALSE, nrow(sce))
  for (cn in defac_cols) {
    vals <- suppressWarnings(as.numeric(rd[[cn]]))
    if (length(vals) != length(marker_gene)) next
    marker_gene <- marker_gene | (is.finite(vals) & (abs(vals - 1.0) > epsilon))
  }
  marker_gene
}

# Precompute masks for dropout, biological zero, and marker-gene stratification.
compute_masks <- function(log_true, log_obs, marker_gene = NULL) {
  epsilon <- 1e-8
  n_genes <- nrow(log_true)
  n_cells <- ncol(log_true)
  marker_gene_mask <- rep(FALSE, n_genes)
  if (!is.null(marker_gene)) {
    marker_gene <- as.logical(marker_gene)
    if (length(marker_gene) != n_genes) {
      stopf("Dimension mismatch: marker_gene length vs log_true genes")
    }
    marker_gene_mask <- marker_gene
  }
  marker_mask <- matrix(marker_gene_mask, nrow = n_genes, ncol = n_cells, byrow = FALSE)
  list(
    biozero = log_true <= epsilon,
    dropout = (log_true > epsilon) & (log_obs <= epsilon),
    non_zero = (log_true > epsilon) & (log_obs > epsilon),
    marker = marker_mask,
    marker_gene = marker_gene_mask
  )
}

pop_sd <- function(x) {
  x <- as.numeric(x)
  if (length(x) == 0) return(NA_real_)
  m <- mean(x)
  sqrt(mean((x - m)^2))
}

compute_gnrmse <- function(log_imp, log_true, gene_mask = NULL, epsilon = 1e-8) {
  diff <- log_true - log_imp
  # Matrices are genes x cells. gNRMSE aggregates gene-wise errors.
  rmse_gene <- sqrt(rowMeans(diff^2))
  sd_true <- apply(log_true, 1, pop_sd)
  denom <- pmax(sd_true, epsilon)
  vals <- rmse_gene / denom
  if (!is.null(gene_mask)) {
    gm <- as.logical(gene_mask)
    if (length(gm) != length(vals)) stopf("gNRMSE gene mask length mismatch")
    vals <- vals[gm]
  }
  vals <- vals[is.finite(vals)]
  if (length(vals) == 0) return(NA_real_)
  mean(vals)
}

compute_corr_err <- function(log_imp, log_true, epsilon = 1e-8) {
  # Correlation is computed across cells, between genes.
  sd_true <- apply(log_true, 1, pop_sd)
  sd_imp <- apply(log_imp, 1, pop_sd)
  keep <- is.finite(sd_true) & is.finite(sd_imp) & (sd_true > epsilon) & (sd_imp > epsilon)
  g_corr <- sum(keep)
  if (g_corr < 2) {
    return(list(corr_err = NA_real_, n_corr_genes = as.integer(g_corr)))
  }

  true_sub <- log_true[keep, , drop = FALSE]
  imp_sub <- log_imp[keep, , drop = FALSE]
  cor_true <- suppressWarnings(stats::cor(t(true_sub), method = "pearson"))
  cor_imp <- suppressWarnings(stats::cor(t(imp_sub), method = "pearson"))
  if (any(!is.finite(cor_true)) || any(!is.finite(cor_imp))) {
    return(list(corr_err = NA_real_, n_corr_genes = as.integer(g_corr)))
  }

  upper_idx <- upper.tri(cor_true, diag = FALSE)
  diffs <- abs(cor_true[upper_idx] - cor_imp[upper_idx])
  if (length(diffs) == 0 || any(!is.finite(diffs))) {
    return(list(corr_err = NA_real_, n_corr_genes = as.integer(g_corr)))
  }
  list(corr_err = mean(diffs), n_corr_genes = as.integer(g_corr))
}

compute_error_metrics <- function(log_imp, log_true, masks) {
  log_imp <- as.matrix(log_imp)
  log_true <- as.matrix(log_true)
  if (!all(dim(log_imp) == dim(log_true))) stopf("Dimension mismatch: log_imp vs log_true")
  if (!all(dim(masks$marker) == dim(log_true))) stopf("Dimension mismatch: marker mask vs log_true")
  if (length(masks$marker_gene) != nrow(log_true)) stopf("Dimension mismatch: marker gene mask vs log_true")

  diff <- log_true - log_imp
  sq_diff <- diff^2
  abs_diff <- abs(diff)

  masked_mean <- function(values, mask) {
    n <- sum(mask)
    if (n <= 0) return(NA_real_)
    mean(values[mask])
  }

  corr_info <- compute_corr_err(log_imp, log_true)

  data.frame(
    mse = mean(sq_diff),
    mse_dropout = masked_mean(sq_diff, masks$dropout),
    mse_biozero = masked_mean(sq_diff, masks$biozero),
    mse_non_zero = masked_mean(sq_diff, masks$non_zero),
    mse_marker = masked_mean(sq_diff, masks$marker),
    mae = mean(abs_diff),
    mae_dropout = masked_mean(abs_diff, masks$dropout),
    mae_biozero = masked_mean(abs_diff, masks$biozero),
    mae_non_zero = masked_mean(abs_diff, masks$non_zero),
    mae_marker = masked_mean(abs_diff, masks$marker),
    gnrmse = compute_gnrmse(log_imp, log_true),
    gnrmse_marker = compute_gnrmse(log_imp, log_true, masks$marker_gene),
    corr_err = corr_info$corr_err,
    n_corr_genes = corr_info$n_corr_genes,
    n_total = length(diff),
    n_dropout = sum(masks$dropout),
    n_biozero = sum(masks$biozero),
    n_non_zero = sum(masks$non_zero),
    n_marker = sum(masks$marker),
    n_marker_genes = sum(masks$marker_gene),
    stringsAsFactors = FALSE
  )
}

summarize_repeats <- function(
  metrics_list,
  runtimes,
  n_total,
  n_dropout,
  n_biozero,
  n_non_zero,
  n_marker,
  n_marker_genes,
  err_msg
) {
  metric_cols <- c(
    "mse", "mse_dropout", "mse_biozero", "mse_non_zero", "mse_marker",
    "mae", "mae_dropout", "mae_biozero", "mae_non_zero", "mae_marker",
    "gnrmse", "gnrmse_marker", "corr_err", "n_corr_genes"
  )
  to_na_if_nan <- function(x) {
    if (is.nan(x)) NA_real_ else x
  }
  if (length(metrics_list) > 0) {
    metrics_df <- do.call(rbind, metrics_list)
    means <- sapply(metric_cols, function(cn) to_na_if_nan(mean(metrics_df[[cn]], na.rm = TRUE)))
    sds <- sapply(
      metric_cols,
      function(cn) if (nrow(metrics_df) > 1) to_na_if_nan(stats::sd(metrics_df[[cn]], na.rm = TRUE)) else 0
    )
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
      mse_marker = means["mse_marker"],
      mse_marker_std = sds["mse_marker"],
      mae = means["mae"],
      mae_std = sds["mae"],
      mae_dropout = means["mae_dropout"],
      mae_dropout_std = sds["mae_dropout"],
      mae_biozero = means["mae_biozero"],
      mae_biozero_std = sds["mae_biozero"],
      mae_non_zero = means["mae_non_zero"],
      mae_non_zero_std = sds["mae_non_zero"],
      mae_marker = means["mae_marker"],
      mae_marker_std = sds["mae_marker"],
      gnrmse = means["gnrmse"],
      gnrmse_std = sds["gnrmse"],
      gnrmse_marker = means["gnrmse_marker"],
      gnrmse_marker_std = sds["gnrmse_marker"],
      corr_err = means["corr_err"],
      corr_err_std = sds["corr_err"],
      n_corr_genes = means["n_corr_genes"],
      n_corr_genes_std = sds["n_corr_genes"],
      runtime_sec = runtime_mean,
      runtime_sec_std = runtime_sd,
      n_repeats = length(runtimes),
      n_total = n_total,
      n_dropout = n_dropout,
      n_biozero = n_biozero,
      n_non_zero = n_non_zero,
      n_marker = n_marker,
      n_marker_genes = n_marker_genes,
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
      mse_marker = NA_real_,
      mse_marker_std = NA_real_,
      mae = NA_real_,
      mae_std = NA_real_,
      mae_dropout = NA_real_,
      mae_dropout_std = NA_real_,
      mae_biozero = NA_real_,
      mae_biozero_std = NA_real_,
      mae_non_zero = NA_real_,
      mae_non_zero_std = NA_real_,
      mae_marker = NA_real_,
      mae_marker_std = NA_real_,
      gnrmse = NA_real_,
      gnrmse_std = NA_real_,
      gnrmse_marker = NA_real_,
      gnrmse_marker_std = NA_real_,
      corr_err = NA_real_,
      corr_err_std = NA_real_,
      n_corr_genes = NA_real_,
      n_corr_genes_std = NA_real_,
      runtime_sec = NA_real_,
      runtime_sec_std = NA_real_,
      n_repeats = 0L,
      n_total = n_total,
      n_dropout = n_dropout,
      n_biozero = n_biozero,
      n_non_zero = n_non_zero,
      n_marker = n_marker,
      n_marker_genes = n_marker_genes,
      error = err_msg,
      stringsAsFactors = FALSE
    )
  }
}

write_method_table <- function(results_df, method) {
  out_path <- file.path(output_dir, paste0(method, "_mse_table.tsv"))
  out_df <- results_df[results_df$method == method, , drop = FALSE]
  out_df$method <- NULL
  col_order <- c(
    "dataset",
    "mse", "mse_std", "mse_dropout", "mse_dropout_std", "mse_biozero", "mse_biozero_std", "mse_non_zero", "mse_non_zero_std", "mse_marker", "mse_marker_std",
    "mae", "mae_std", "mae_dropout", "mae_dropout_std", "mae_biozero", "mae_biozero_std", "mae_non_zero", "mae_non_zero_std", "mae_marker", "mae_marker_std",
    "gnrmse", "gnrmse_std", "gnrmse_marker", "gnrmse_marker_std", "corr_err", "corr_err_std", "n_corr_genes", "n_corr_genes_std",
    "runtime_sec", "runtime_sec_std", "n_repeats",
    "n_total", "n_dropout", "n_biozero", "n_non_zero", "n_marker", "n_marker_genes",
    "error"
  )
  missing_cols <- setdiff(col_order, colnames(out_df))
  for (cn in missing_cols) out_df[[cn]] <- NA
  out_df <- out_df[, col_order, drop = FALSE]

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

dataset_name_from_path <- function(path) {
  stem <- tools::file_path_sans_ext(basename(path))
  if (tolower(stem) != "sce") return(stem)

  parent <- basename(dirname(path))
  grandparent <- basename(dirname(dirname(path)))
  if (grandparent %in% c("test", "tune") && nzchar(parent)) return(parent)
  if (startsWith(parent, "n") && nzchar(grandparent)) return(paste0(grandparent, "_", parent))
  if (nzchar(parent)) return(parent)
  stem
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
  dataset_name <- dataset_name_from_path(input_file)
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
  marker_gene <- extract_marker_gene_mask(sce)

  if (!all(dim(counts) == dim(log_obs)) || !all(dim(log_true) == dim(log_obs))) {
    stopf("[%s] Assay dimension mismatch among counts/logcounts/logTrueCounts.", dataset_name)
  }

  norm_info <- get_normalization_info(sce)
  target_sum <- as.numeric(norm_info$target_sum)

  masks <- compute_masks(log_true, log_obs, marker_gene = marker_gene)
  n_total <- length(log_true)
  n_dropout <- sum(masks$dropout)
  n_biozero <- sum(masks$biozero)
  n_non_zero <- sum(masks$non_zero)
  n_marker <- sum(masks$marker)
  n_marker_genes <- sum(masks$marker_gene)

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
        compute_error_metrics(log_imp, log_true, masks)
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
        n_marker,
        n_marker_genes,
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
    # SAVER drops zero-count cells internally; pre-filter zero-expression cells.
    lib_obs <- Matrix::colSums(counts)
    nonzero_cells <- lib_obs > 0
    if (!all(nonzero_cells)) {
      cat(sprintf("  [saver] Dropping %d zero-expression cells.\n", sum(!nonzero_cells)))
    }
    saver_cells <- nonzero_cells
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

      saver_ncores <- min(ncores, saver_default_cores)
      if (saver_ncores < 1) saver_ncores <- 1
      message(sprintf("Running SAVER with %d worker(s)", saver_ncores))
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
            saver_est_full <- matrix(0, nrow = nrow(counts), ncol = ncol(counts), dimnames = dimnames(counts))
            saver_est_full[, saver_cells] <- saver_counts
            if (i == 1) save_imputed(dataset_name, saver_est_full, "saver")

            log_imp <- log_obs
            log_imp[, saver_cells] <- normalize_counts_to_logcounts(
              saver_counts,
              target_sum
            )
            compute_error_metrics(log_imp, log_true, masks)
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
        n_marker,
        n_marker_genes,
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

    cc_ncores <- min(ncores, 8)
    if (ncores > 8) {
      message(sprintf("ccImpute is capped at 8 cores; using %d instead of %d.", cc_ncores, ncores))
    }

    bpp <- BiocParallel::SerialParam()
    if (cc_ncores > 1) {
      bpp <- tryCatch(
        BiocParallel::MulticoreParam(workers = cc_ncores),
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
        compute_error_metrics(log_imp, log_true, masks)
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
        n_marker,
        n_marker_genes,
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
