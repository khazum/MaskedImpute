#!/usr/bin/env Rscript

# Description:
# This script processes a directory of pre-normalized and pre-log-transformed
# SingleCellExperiment (SCE) objects.
# For each SCE, it selects the top N highly variable genes (HVGs) using an
# AutoClass-style score: variance(non-zero expression) / mean(non-zero expression).
# No additional normalization or log transformation is applied.

# --- Load required libraries ---
suppressPackageStartupMessages(library(argparse))
suppressPackageStartupMessages(library(SingleCellExperiment))
suppressPackageStartupMessages(library(Matrix))

# --- Define Command-Line Arguments ---
parser <- ArgumentParser(
  description = paste(
    "Select top HVGs from pre-normalized, pre-log-transformed",
    "SingleCellExperiment objects without re-normalization."
  )
)
parser$add_argument("-i", "--input_dir", type = "character", required = TRUE,
                    help = "Path to the directory containing input RDS files.")
parser$add_argument("-o", "--output_dir", type = "character", required = TRUE,
                    help = "Path to the directory where output RDS files will be saved.")
parser$add_argument("-n", "--n_genes", type = "integer", default = 1000,
                    help = "Number of top HVGs to select [default: %(default)s].")
args <- parser$parse_args()

if (is.na(args$n_genes) || args$n_genes < 1L) {
  stop("--n_genes must be a positive integer.")
}

row_sums_any <- function(x) {
  if (inherits(x, "Matrix")) {
    Matrix::rowSums(x)
  } else {
    rowSums(x)
  }
}

# Mimics the reference HVG logic:
# score(gene) = var(non-zero gene expression) / mean(non-zero gene expression)
# where variance uses population variance (ddof = 0).
find_hv_genes <- function(expr_mat, top = 1000L) {
  n_genes <- nrow(expr_mat)
  if (is.null(n_genes) || n_genes == 0L || top <= 0L) {
    return(integer(0))
  }

  nz_counts <- row_sums_any(expr_mat != 0)
  nz_sums <- row_sums_any(expr_mat)
  nz_sq_sums <- row_sums_any(expr_mat * expr_mat)

  mu <- rep(NA_real_, n_genes)
  valid_nz <- nz_counts > 0
  mu[valid_nz] <- nz_sums[valid_nz] / nz_counts[valid_nz]

  var_nz <- rep(NA_real_, n_genes)
  var_nz[valid_nz] <- nz_sq_sums[valid_nz] / nz_counts[valid_nz] - mu[valid_nz]^2
  var_nz[var_nz < 0] <- 0

  hv_score <- var_nz / mu
  hv_score[!is.finite(hv_score) | mu <= 0] <- -Inf

  valid_score_idx <- which(is.finite(hv_score) & hv_score > -Inf)
  if (length(valid_score_idx) == 0L) {
    return(integer(0))
  }

  ranked <- valid_score_idx[order(hv_score[valid_score_idx], decreasing = TRUE)]
  ranked[seq_len(min(as.integer(top), length(ranked)))]
}

select_expression_assay <- function(sce) {
  preferred <- c("logcounts", "logTrueCounts", "perfect_logcounts", "counts", "TrueCounts")
  available <- intersect(preferred, assayNames(sce))
  if (length(available) == 0L) {
    stop(
      "No compatible assay found. Expected one of: ",
      paste(preferred, collapse = ", ")
    )
  }
  available[[1]]
}

# --- Main Processing Logic ---
if (!dir.exists(args$input_dir)) {
  stop("Input directory does not exist: ", args$input_dir)
}

rds_files <- list.files(
  path = args$input_dir,
  pattern = "\\.rds$",
  full.names = TRUE,
  ignore.case = TRUE
)
if (length(rds_files) == 0L) {
  stop("No .rds files found in the specified input directory.")
}

if (!dir.exists(args$output_dir)) {
  message("Output directory does not exist. Creating it now: ", args$output_dir)
  dir.create(args$output_dir, recursive = TRUE)
}

message(paste("\nFound", length(rds_files), "RDS file(s) to process."))
for (file_path in rds_files) {
  base_name <- tools::file_path_sans_ext(basename(file_path))
  message(paste("\n--- Processing:", base_name, "---"))

  tryCatch({
    message("  -> Reading RDS file...")
    sce <- readRDS(file_path)

    if (!inherits(sce, "SingleCellExperiment")) {
      warning(paste("Skipping", base_name, "as it is not a SingleCellExperiment object."))
      next
    }
    if (nrow(sce) == 0L || ncol(sce) == 0L) {
      warning(paste("Skipping", base_name, "- empty SCE object."))
      next
    }

    assay_name <- select_expression_assay(sce)
    message(paste("  -> Using assay for HVG scoring:", assay_name))
    expr_mat <- assay(sce, assay_name)

    n_target <- min(args$n_genes, nrow(sce))
    if (n_target < args$n_genes) {
      message(sprintf(
        "  -> Requested %d HVGs but only %d genes available; selecting %d.",
        args$n_genes, nrow(sce), n_target
      ))
    }

    message(paste(
      "  -> Selecting top", n_target,
      "HVGs by var(non-zero)/mean(non-zero) without re-normalization..."
    ))
    top_idx <- find_hv_genes(expr_mat, top = n_target)
    if (length(top_idx) == 0L) {
      warning(paste("Skipping", base_name, "- no valid HVGs found."))
      next
    }
    if (length(top_idx) < n_target) {
      message(sprintf(
        "  -> Only %d genes had valid non-zero statistics; using all of them.",
        length(top_idx)
      ))
    }

    sce_subset <- sce[top_idx, , drop = FALSE]
    message(paste("  -> Subset object to", nrow(sce_subset), "HVGs."))

    # Ensure all assays are dense (avoid sparse matrix issues downstream).
    for (assay_name_i in assayNames(sce_subset)) {
      mat <- assay(sce_subset, assay_name_i)
      if (inherits(mat, "sparseMatrix") || inherits(mat, "Matrix")) {
        assay(sce_subset, assay_name_i) <- as.matrix(mat)
      }
    }

    output_filename <- paste0(base_name, "_top", args$n_genes, "hvg.rds")
    output_path <- file.path(args$output_dir, output_filename)
    saveRDS(sce_subset, file = output_path)
    message(paste("  -> Successfully saved subsetted object to:", output_path))

  }, error = function(e) {
    message(paste("\n[ERROR] Failed to process", base_name, ":", e$message, "\n"))
  })
}

message("\n--- All processing complete. ---")
