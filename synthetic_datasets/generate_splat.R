# generate_splat.R
# Purpose: Generate scRNA-seq simulations (Full SCE).
# Required packages: splatter, SingleCellExperiment, scater, ggplot2, cowplot, Rtsne,
# Matrix, ragg.
#      Counts and TrueCounts are normalized using TrueCounts-derived library sizes
#      and a shared scale factor (median TrueCounts library size).
#      Outputs per-cell-count folders, simulates 1500 genes, and filters low-expression genes.
#      Output forces Dense Matrices & Integer Dropout for python compatibility.

# --- 1) Configuration & Setup ---
OUT_DIR <- "rds_splat_output"
#CELL_COUNTS <- c(1000, 5000, 10000, 15000, 20000, 25000, 50000, 75000, 100000)
CELL_COUNTS <- c(50000, 75000, 100000)
N_GENES <- 1100
MIN_CELLS_EXPRESSED <- 3
SEED <- 42

if (!dir.exists(OUT_DIR)) dir.create(OUT_DIR, recursive = TRUE)

require_pkg <- function(pkg) {
  if (requireNamespace(pkg, quietly = TRUE)) return(invisible(TRUE))
  stop(sprintf("Missing required package '%s'. See header for dependencies.", pkg), call. = FALSE)
}

load_pkg <- function(pkg) {
  suppressPackageStartupMessages(library(pkg, character.only = TRUE))
}

required_pkgs <- c("splatter", "SingleCellExperiment")#, "scater", "ggplot2", "cowplot",
#                   "Rtsne", "Matrix", "ragg")
invisible(lapply(required_pkgs, require_pkg))
invisible(lapply(required_pkgs, load_pkg))

cat(sprintf("Output directory: %s\n", OUT_DIR))
cat(sprintf("Cell counts: %s\n", paste(CELL_COUNTS, collapse = ", ")))
cat(sprintf("Simulated genes: %d | Min cells expressed: %d\n", N_GENES, MIN_CELLS_EXPRESSED))

# --- 2) Helper Functions ---

# Modified: Normalize both 'counts' and 'TrueCounts' using TrueCounts-derived
#           library sizes and a shared scale factor (median TrueCounts library size).
apply_normalization <- function(sce) {
  stopifnot("TrueCounts" %in% assayNames(sce), "counts" %in% assayNames(sce))

  # --- Library sizes from TrueCounts ---
  lib.true <- colSums(assay(sce, "TrueCounts"))
  denom.true <- ifelse(lib.true > 0, lib.true, 1)
  scale_factor <- stats::median(lib.true[lib.true > 0])
  if (!is.finite(scale_factor) || scale_factor <= 0) scale_factor <- 1
  cat(sprintf("   > Normalization scale factor (median TrueCounts lib size): %.4f\n", scale_factor))

  # --- A. Normalize Noisy Counts (Using TrueCounts-derived factors) ---
  noisy_norm <- t(t(assay(sce, "counts")) / denom.true * scale_factor)
  assay(sce, "logcounts") <- log2(1 + noisy_norm)
  
  # --- B. Normalize True Counts (Using TrueCounts-derived factors) ---
  perfect_norm <- t(t(assay(sce, "TrueCounts")) / denom.true * scale_factor)
  assay(sce, "logTrueCounts") <- log2(1 + perfect_norm)

  # Store normalization info for downstream use
  colData(sce)$libSizeTrueCounts <- lib.true
  colData(sce)$scaleFactorTrueCounts <- scale_factor
  metadata(sce)$normalization <- list(
    library_sizes = lib.true,
    scale_factor = scale_factor
  )
  
  return(sce)
}

# Split total cells across batches with remainder assigned to early batches.
split_batches <- function(total_cells, n_batches) {
  base <- floor(total_cells / n_batches)
  remainder <- total_cells - base * n_batches
  counts <- rep(base, n_batches)
  if (remainder > 0) {
    counts[seq_len(remainder)] <- counts[seq_len(remainder)] + 1
  }
  return(counts)
}

# Simulate a SingleCellExperiment with provided parameters.
simulate_sce <- function(base_params, config, batch_cells, n_genes, dropout_type,
                         current_mid, current_shape, rm_batch_effect) {
  params <- setParams(
    base_params,
    nGenes = n_genes,
    batchCells = batch_cells,
    group.prob = config$props,
    de.prob = 0.1,
    dropout.mid = current_mid,
    dropout.shape = current_shape,
    dropout.type = dropout_type
  )
  splatSimulate(params, method = "groups", verbose = FALSE, batch.rmEffect = rm_batch_effect)
}

# Filter genes expressed in fewer than min_cells in counts and apply the same
# gene set across all assays.
filter_low_expression_genes <- function(sce, min_cells) {
  stopifnot("counts" %in% assayNames(sce))
  gene_keep <- rowSums(assay(sce, "counts") > 0) >= min_cells
  cat(sprintf("   > Genes retained after filter: %d -> %d\n", nrow(sce), sum(gene_keep)))
  
  # Row subsetting applies the filter across all assays (counts, TrueCounts,
  # logcounts, logTrueCounts) and associated row metadata.
  sce <- sce[gene_keep, , drop = FALSE]
  
  core_assays <- intersect(c("counts", "TrueCounts", "logcounts", "logTrueCounts"), assayNames(sce))
  mismatch <- core_assays[
    vapply(core_assays, function(nm) nrow(assay(sce, nm)) != nrow(sce), logical(1))
  ]
  if (length(mismatch) > 0) {
    stop(sprintf("Assay row mismatch after filtering: %s", paste(mismatch, collapse = ", ")))
  }
  
  sce
}

# Run t-SNE on noisy and true log counts.
run_tsne <- function(sce) {
  perp <- max(5, min(30, floor((ncol(sce) - 1) / 3)))
  sce <- runTSNE(sce, exprs_values = "logcounts", name = "TSNE_noisy", perplexity = perp)
  sce <- runTSNE(sce, exprs_values = "logTrueCounts", name = "TSNE_logTrue", perplexity = perp)
  sce
}

# Convert assays to base matrices and ensure Dropout is integer.
coerce_assays_to_dense <- function(sce) {
  for (nm in assayNames(sce)) {
    mat <- assay(sce, nm)
    
    # Convert Matrix classes (sparse or dense) to base matrix.
    if (inherits(mat, "Matrix")) {
      mat <- as.matrix(mat)
    }
    
    # Fix Dropout: Logical -> Integer (0/1).
    if (nm == "Dropout" && !is.integer(mat)) {
      mat <- matrix(as.integer(mat), nrow=nrow(mat), ncol=ncol(mat), dimnames=dimnames(mat))
    }
    
    assay(sce, nm) <- mat
  }
  sce
}

save_dimred_plot <- function(sce, method, filename, title_str, subtitle_str) {
  noisy_dim   <- paste0(method, "_noisy")
  true_dim    <- paste0(method, "_logTrue")
  
  p_true <- plotReducedDim(sce, dimred = true_dim, colour_by = "Group") + 
    ggtitle(paste(method, "(True Signal)")) + theme(legend.position = "none")
  
  p_noisy <- plotReducedDim(sce, dimred = noisy_dim, colour_by = "Group") + 
    ggtitle(paste(method, "(Noisy Counts)")) + 
    labs(subtitle = subtitle_str) + 
    theme(legend.position = "bottom", plot.subtitle = element_text(size = 8))
  
  final_grid <- plot_grid(
    ggdraw() + draw_label(title_str, fontface = 'bold', x = 0.01, hjust = 0),
    plot_grid(p_true, p_noisy, ncol = 2),
    ncol = 1, rel_heights = c(0.1, 1)
  )
  
  # Use ragg to avoid system cairo/tiff dependencies.
  ggsave(filename, final_grid, width = 10, height = 5, bg = "white", device = ragg::agg_png)
  cat(sprintf("   > Saved plot: %s\n", filename))
}

# --- 3) Simulation Parameters ---
sim_configs <- list(
  dataset_2_types_equal = list(
    name = "2 Types, Equal", groups = 2, props = c(0.5, 0.5)
  ),
  dataset_3_types_unequal = list(
    name = "3 Types, Unequal", groups = 3, props = c(0.6, 0.3, 0.1)
  ),
  dataset_multibatch_dropout = list(
    name = "3 Types, 3 Batches (Batch Dropout)", groups = 3, props = c(0.4, 0.3, 0.3),
    n_batches = 3, dropout.type = "batch", batch.rmEffect = TRUE
  ),
  dataset_4_types_rare = list(
    name = "4 Types, Rare Pop", groups = 4, props = c(0.5, 0.25, 0.20, 0.05)
  )
)

set.seed(SEED)
base_params <- setParams(newSplatParams(), seed = SEED)
mid_range   <- c(1, 5)
shape_range <- c(-1.5, -0.5)

# --- 4) Main Loop ---
for (sim_id in names(sim_configs)) {
  config <- sim_configs[[sim_id]]
  cat(paste0("\nProcessing config: ", sim_id, " ...\n"))
  
  dropout_type    <- if (!is.null(config$dropout.type)) config$dropout.type else "group"
  rm_batch_effect <- isTRUE(config$batch.rmEffect)
  n_batches       <- if (!is.null(config$n_batches)) config$n_batches else 1
  
  dropout_dim   <- if (dropout_type == "group") config$groups else n_batches
  current_mid   <- runif(dropout_dim, mid_range[1], mid_range[2])
  current_shape <- runif(dropout_dim, shape_range[1], shape_range[2])
  
  for (cell_count in CELL_COUNTS) {
    cell_dir <- file.path(OUT_DIR, paste0("cells_", cell_count))
    if (!dir.exists(cell_dir)) dir.create(cell_dir, recursive = TRUE)
    cat(paste0("  Cell count: ", cell_count, " ...\n"))
    
    batch_cells <- if (n_batches > 1) split_batches(cell_count, n_batches) else cell_count
    
    # 1. Simulate with a fixed gene count.
    sce <- simulate_sce(
      base_params,
      config,
      batch_cells,
      N_GENES,
      dropout_type,
      current_mid,
      current_shape,
      rm_batch_effect
    )
    cat(sprintf("   > Simulated genes: %d\n", nrow(sce)))
    
    # 2. Filter genes expressed in fewer than MIN_CELLS_EXPRESSED cells.
    sce <- filter_low_expression_genes(sce, MIN_CELLS_EXPRESSED)
    
    # 3. Normalize (Counts and TrueCounts use TrueCounts-derived library sizes).
    sce <- apply_normalization(sce)
    
    # 4. Dimensionality reduction (t-SNE only).
#    sce <- run_tsne(sce)
    
#    # 5. Visualize.
#    sub_str <- sprintf("Cells: %d | Batches: %d | Drop: %s | Mid: %s | Shp: %s | Genes: %d",
#                       cell_count, n_batches, dropout_type,
#                       paste(round(current_mid, 2), collapse=","), 
#                       paste(round(current_shape, 2), collapse=","),
#                       nrow(sce))
    
#    save_dimred_plot(sce, "TSNE", file.path(cell_dir, paste0(sim_id, "_TSNE.png")), 
#                     paste("t-SNE:", config$name), sub_str)
    
    # 6. Prepare for save (dense assays + integer Dropout).
    sce <- coerce_assays_to_dense(sce)
    
    # Save Full Object
    saveRDS(sce, file = file.path(cell_dir, paste0(sim_id, ".rds")))
    cat(sprintf("   > Saved RDS (Dense Matrices): %s\n", file.path(cell_dir, paste0(sim_id, ".rds"))))
  }
}

cat("\nAll simulations completed successfully.\n")
