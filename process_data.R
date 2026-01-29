#!/usr/bin/env Rscript

# Description:
# This script processes a directory of SingleCellExperiment (SCE) objects stored as .rds files.
# For each SCE object, it filters genes expressed in at least 3 cells, computes
# deconvolution size factors with scran, applies log-normalization, and selects
# a specified number of highly variable genes (HVGs).
# The resulting subsetted SCE object is saved to an output directory.
#
# Author: Gemini
# Date: 2025-09-17

# --- Load required libraries ---
# Use suppressPackageStartupMessages to keep the console output clean.
suppressPackageStartupMessages(library(argparse))
suppressPackageStartupMessages(library(scran))
suppressPackageStartupMessages(library(SingleCellExperiment))
suppressPackageStartupMessages(library(BiocSingular))

# --- Define Command-Line Arguments ---
parser <- ArgumentParser(description = "Normalize and find HVGs in SingleCellExperiment objects using scran deconvolution size factors.")

parser$add_argument("-i", "--input_dir", type = "character", required = TRUE,
                    help = "Path to the directory containing input RDS files.")

parser$add_argument("-o", "--output_dir", type = "character", required = TRUE,
                    help = "Path to the directory where output RDS files will be saved.")

parser$add_argument("-n", "--n_genes", type = "integer", default = 1000,
                    help = "Number of top highly variable genes to select [default: %(default)s].")

parser$add_argument("--overwrite", action = "store_true", default = FALSE,
                    help = "Overwrite the original RDS files in place.")

# Parse the arguments from the command line
args <- parser$parse_args()


# --- Main Processing Logic ---

# 1. Validate input directory and find RDS files
if (!dir.exists(args$input_dir)) {
  stop("Input directory does not exist: ", args$input_dir)
}

# Find all files ending in .rds (case-insensitive)
rds_files <- list.files(path = args$input_dir, pattern = "\\.rds$", full.names = TRUE, ignore.case = TRUE)

if (length(rds_files) == 0) {
  stop("No .rds files found in the specified input directory.")
}

# 2. Create the output directory if it doesn't exist
if (!dir.exists(args$output_dir)) {
  message("Output directory does not exist. Creating it now: ", args$output_dir)
  dir.create(args$output_dir, recursive = TRUE)
}

# 3. Loop through each file and process it
message(paste("\nFound", length(rds_files), "RDS file(s) to process."))

for (file_path in rds_files) {
  base_name <- tools::file_path_sans_ext(basename(file_path))
  message(paste("\n--- Processing:", base_name, "---"))

  tryCatch({
    # Read the SingleCellExperiment object
    message("  -> Reading RDS file...")
    sce <- readRDS(file_path)

    # Ensure the object is a SingleCellExperiment
    if (!inherits(sce, "SingleCellExperiment")) {
        warning(paste("Skipping", base_name, "as it is not a SingleCellExperiment object."))
        next
    }

    # --- Filtering ---
    message("  -> Filtering genes expressed in at least 3 cells...")
    if (!"counts" %in% assayNames(sce)) {
        stop("The SCE object does not contain a 'counts' assay.")
    }
    counts_mat <- counts(sce)
    # Filter genes expressed in at least 3 cells
    expressed_cells <- Matrix::rowSums(counts_mat > 0)
    keep_genes <- expressed_cells >= 3
    if (!any(keep_genes)) {
        stop("No genes expressed in at least 3 cells after filtering.")
    }
    sce <- sce[keep_genes, ]
    counts_mat <- counts(sce)

    # --- Normalization ---
    message("  -> Computing size factors with scran deconvolution...")
    lib_sizes <- colSums(counts_mat)
    zero_cells <- lib_sizes <= 0
    if (any(zero_cells)) {
        message(sprintf("  -> Dropping %d cells with zero library size.", sum(zero_cells)))
        sce <- sce[, !zero_cells, drop = FALSE]
        counts_mat <- counts(sce)
        lib_sizes <- colSums(counts_mat)
    }
    if (ncol(sce) < 2) {
        stop("Too few cells after filtering; cannot normalize.")
    }

    n_cells <- ncol(sce)
    n_genes <- nrow(sce)
    min_dim <- min(n_cells, n_genes)
    min_cluster_size <- min(100L, max(2L, floor(n_cells / 2L)))

    clusters <- NULL
    if (n_cells >= 10 && min_dim >= 3) {
        d_pca <- min(50L, max(1L, min_dim - 1L))
        bsparam <- if (min_dim < 200L) BiocSingular::ExactParam() else BiocSingular::IrlbaParam()
        clusters <- scran::quickCluster(
            sce,
            min.mean = 0.1,
            min.size = min_cluster_size,
            d = d_pca,
            BSPARAM = bsparam
        )
    } else {
        message("  -> Skipping quickCluster (too few cells/genes); using single cluster.")
        clusters <- factor(rep(1L, n_cells))
    }
    sce <- scran::computeSumFactors(sce, clusters = clusters)
    size_factors <- sizeFactors(sce)
    if (any(!is.finite(size_factors) | size_factors <= 0)) {
        stop("Invalid size factors computed; cannot normalize.")
    }

    message("  -> Applying log-normalization...")
    # Normalize counts by size factors, then log2(1 + x)
    norm_counts <- t(t(counts_mat) / size_factors)
    assay(sce, "logcounts", withDimnames = FALSE) <- log2(norm_counts + 1)
    # Store normalization metadata (counts-derived) for downstream tools
    scale_factor <- stats::median(lib_sizes[lib_sizes > 0])
    colData(sce)$libSizeTrueCounts <- lib_sizes
    colData(sce)$sizeFactorTrueCounts <- size_factors
    colData(sce)$scaleFactorTrueCounts <- scale_factor
    metadata(sce)$normalization <- list(
      method = "scran::computeSumFactors",
      library_sizes = lib_sizes,
      size_factors = size_factors,
      scale_factor = scale_factor,
      log_base = 2,
      pseudo_count = 1
    )
    message("  -> Normalization complete.")

    # --- Highly Variable Gene (HVG) Selection ---
    n_hvg <- min(args$n_genes, nrow(sce))
    if (n_hvg < args$n_genes) {
        message(sprintf("  -> Requested %d HVGs but only %d genes available; using %d.", args$n_genes, nrow(sce), n_hvg))
    }
    message(paste("  -> Modeling gene variance to find top", n_hvg, "HVGs..."))
    gene_var <- modelGeneVar(sce)
    top_hvgs <- getTopHVGs(gene_var, n = n_hvg)
    message("  -> HVG selection complete.")

    # --- Subsetting and Saving ---
    message("  -> Subsetting the object to keep only HVGs...")
    # Create a new SCE object containing only the top HVGs
    sce_hvg <- sce[top_hvgs, ]

    # Define the output file path
    same_dir <- normalizePath(args$input_dir) == normalizePath(args$output_dir)
    overwrite <- isTRUE(args$overwrite) || same_dir
    if (overwrite) {
        output_path <- file_path
        message("  -> Overwriting original file in place.")
    } else {
        output_filename <- paste0(base_name, "_top", args$n_genes, "hvg.rds")
        output_path <- file.path(args$output_dir, output_filename)
    }

    # Save the new object
    saveRDS(sce_hvg, file = output_path)
    message(paste("  -> Successfully saved subsetted object to:", output_path))

  }, error = function(e) {
    # Handle and report any errors that occur during processing
    message(paste("\n[ERROR] Failed to process", base_name, ":", e$message, "\n"))
  })
}

message("\n--- All processing complete. ---")
