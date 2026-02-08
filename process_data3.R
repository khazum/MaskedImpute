#!/usr/bin/env Rscript

# Description:
# Process SingleCellExperiment (SCE) .rds files that are already CPM-normalized
# and log-transformed. The script does not re-normalize data.
#
# It selects top HVGs with scran::modelGeneVar/getTopHVGs on existing logcounts.
# It also stores per-cell CPM library-size metadata from full-gene counts prior
# to HVG reduction so clustering/DCA code can reuse original library sizes.

suppressPackageStartupMessages(library(argparse))
suppressPackageStartupMessages(library(SingleCellExperiment))
suppressPackageStartupMessages(library(scran))
suppressPackageStartupMessages(library(Matrix))

parser <- ArgumentParser(
  description = paste(
    "Select top HVGs from already CPM-normalized/log-transformed SCE objects",
    "using modelGeneVar, without re-normalization."
  )
)
parser$add_argument("-i", "--input_dir", type = "character", required = TRUE,
                    help = "Directory containing input .rds files.")
parser$add_argument("-o", "--output_dir", type = "character", required = TRUE,
                    help = "Directory to write processed .rds files.")
parser$add_argument("-n", "--n_genes", type = "integer", default = 1000,
                    help = "Number of top HVGs to keep [default: %(default)s].")
args <- parser$parse_args()

if (is.na(args$n_genes) || args$n_genes < 1L) {
  stop("--n_genes must be a positive integer.")
}

ensure_required_assays <- function(sce) {
  if (!"logcounts" %in% assayNames(sce)) {
    stop("Missing required 'logcounts' assay (expected pre-log-transformed input).")
  }
  if (!"counts" %in% assayNames(sce)) {
    stop("Missing required 'counts' assay (CPM source for library sizes).")
  }
  sce
}

set_cpm_library_metadata <- function(sce, lib_sizes_full) {
  if (length(lib_sizes_full) != ncol(sce)) {
    stop(sprintf("Library size length (%d) does not match cells (%d).",
                 length(lib_sizes_full), ncol(sce)))
  }

  lib_sizes_full <- as.numeric(lib_sizes_full)
  lib_sizes_full <- ifelse(is.finite(lib_sizes_full) & lib_sizes_full >= 0, lib_sizes_full, 0)

  finite_lib <- lib_sizes_full[lib_sizes_full > 0]
  scale_factor <- if (length(finite_lib) > 0L) stats::median(finite_lib) else 1
  if (!is.finite(scale_factor) || scale_factor <= 0) scale_factor <- 1

  size_factors <- lib_sizes_full / scale_factor
  size_factors <- ifelse(is.finite(size_factors) & size_factors > 0, size_factors, 1)

  # Keep legacy key names for downstream readers, but values come from counts CPM.
  colData(sce)$libSizeTrueCounts <- lib_sizes_full
  colData(sce)$scaleFactorTrueCounts <- rep(scale_factor, ncol(sce))
  colData(sce)$sizeFactorTrueCounts <- size_factors
  sizeFactors(sce) <- size_factors

  metadata(sce)$normalization <- list(
    method = "provided_cpm",
    library_sizes = lib_sizes_full,
    size_factors = size_factors,
    scale_factor = scale_factor,
    log_base = 2,
    pseudo_count = 1
  )

  sce
}

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

message(sprintf("\nFound %d RDS file(s) to process.", length(rds_files)))
for (file_path in rds_files) {
  base_name <- tools::file_path_sans_ext(basename(file_path))
  message(sprintf("\n--- Processing: %s ---", base_name))

  tryCatch({
    message("  -> Reading RDS file...")
    sce <- readRDS(file_path)
    if (!inherits(sce, "SingleCellExperiment")) {
      warning(sprintf("Skipping %s: object is not a SingleCellExperiment.", base_name))
      next
    }
    if (nrow(sce) == 0L || ncol(sce) == 0L) {
      warning(sprintf("Skipping %s: empty SCE object.", base_name))
      next
    }

    message("  -> Using existing CPM/logcounts normalization (no re-normalization).")
    sce <- ensure_required_assays(sce)
    full_lib_sizes <- as.numeric(Matrix::colSums(assay(sce, "counts")))
    sce <- set_cpm_library_metadata(sce, full_lib_sizes)

    n_hvg <- min(args$n_genes, nrow(sce))
    if (n_hvg < args$n_genes) {
      message(sprintf("  -> Requested %d HVGs but only %d genes available; selecting %d.",
                      args$n_genes, nrow(sce), n_hvg))
    }

    message(sprintf("  -> Running modelGeneVar() and selecting top %d HVGs...", n_hvg))
    # Disable lowess smoothing to avoid tie-collapse warnings on heavily tied means.
    gene_var <- scran::modelGeneVar(sce, lowess = FALSE)
    top_hvgs <- scran::getTopHVGs(gene_var, n = n_hvg)
    if (length(top_hvgs) == 0L) {
      stop("No HVGs returned by getTopHVGs.")
    }

    sce_hvg <- sce[top_hvgs, , drop = FALSE]
    message(sprintf("  -> Subset object to %d HVGs.", nrow(sce_hvg)))

    # Ensure assays are dense for downstream compatibility.
    for (assay_name_i in assayNames(sce_hvg)) {
      mat <- assay(sce_hvg, assay_name_i)
      if (inherits(mat, "sparseMatrix") || inherits(mat, "Matrix")) {
        assay(sce_hvg, assay_name_i) <- as.matrix(mat)
      }
    }

    output_filename <- paste0(base_name, "_top", args$n_genes, "hvg.rds")
    output_path <- file.path(args$output_dir, output_filename)
    saveRDS(sce_hvg, file = output_path)
    message(paste("  -> Successfully saved subsetted object to:", output_path))

  }, error = function(e) {
    message(sprintf("\n[ERROR] Failed to process %s: %s\n", base_name, e$message))
  })
}

message("\n--- All processing complete. ---")
