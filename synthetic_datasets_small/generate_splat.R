# generate_splat.R
# Purpose: Generate scRNA-seq simulations (Full SCE).
# Fix: Counts are normalized by size factors derived from COUNTS (realistically), 
#      rather than TrueCounts.
#      Output forces Dense Matrices & Integer Dropout for python compatibility.

# --- 1) Configuration & Setup ---
OUT_DIR <- "rds_splat_output"
options(repos = c(CRAN = "https://cran.rstudio.com/"))

if (!dir.exists(OUT_DIR)) dir.create(OUT_DIR, recursive = TRUE)

required_pkgs <- c("splatter", "SingleCellExperiment", "scuttle", "scater", 
                   "ggplot2", "cowplot", "Rtsne", "BiocGenerics", "S4Vectors")

invisible(lapply(required_pkgs, function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager")
    BiocManager::install(pkg, ask = FALSE, update = FALSE)
  }
  suppressPackageStartupMessages(library(pkg, character.only = TRUE))
}))

cat(sprintf("Output directory: %s\n", OUT_DIR))

# --- 2) Helper Functions ---

# Modified: Normalize 'counts' using size factors from 'counts' (Real world scenario)
#           Normalize 'TrueCounts' using size factors from 'TrueCounts' (Ideal scenario)
apply_normalization <- function(sce) {
  stopifnot("TrueCounts" %in% assayNames(sce), "counts" %in% assayNames(sce))
  
  # --- A. Normalize Noisy Counts (Using counts-derived factors) ---
  lib.noisy <- colSums(assay(sce, "counts"))
  denom.noisy <- ifelse(lib.noisy > 0, lib.noisy, 1)
  med.noisy <- stats::median(lib.noisy[lib.noisy > 0])
  if (!is.finite(med.noisy) || med.noisy <= 0) med.noisy <- 1
  
  noisy_norm <- t(t(assay(sce, "counts")) / denom.noisy * med.noisy)
  assay(sce, "logcounts") <- log2(1 + noisy_norm)
  
  # --- B. Normalize True Counts (Using TrueCounts-derived factors) ---
  # We normalize TrueCounts by its own depth so it remains comparable in magnitude 
  # to the normalized counts (assuming similar median depths).
  lib.true <- colSums(assay(sce, "TrueCounts"))
  denom.true <- ifelse(lib.true > 0, lib.true, 1)
  med.true <- stats::median(lib.true[lib.true > 0])
  if (!is.finite(med.true) || med.true <= 0) med.true <- 1
  
  perfect_norm <- t(t(assay(sce, "TrueCounts")) / denom.true * med.true)
  assay(sce, "logTrueCounts") <- log2(1 + perfect_norm)
  
  return(sce)
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
  
  ggsave(filename, final_grid, width = 10, height = 5, bg = "white")
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
    batchCells = c(100, 100, 100), dropout.type = "batch", batch.rmEffect = TRUE
  ),
  dataset_4_types_rare = list(
    name = "4 Types, Rare Pop", groups = 4, props = c(0.5, 0.25, 0.20, 0.05)
  )
)

base_params <- setParams(newSplatParams(), seed = 42)
mid_range   <- c(1, 5)
shape_range <- c(-1.5, -0.5)

# --- 4) Main Loop ---
for (sim_id in names(sim_configs)) {
  config <- sim_configs[[sim_id]]
  cat(paste0("\nProcessing: ", sim_id, " ...\n"))
  
  # Config
  batch_cells     <- if (!is.null(config$batchCells)) config$batchCells else 100
  n_batches       <- length(batch_cells)
  dropout_type    <- if (!is.null(config$dropout.type)) config$dropout.type else "group"
  rm_batch_effect <- isTRUE(config$batch.rmEffect)
  
  dropout_dim   <- if (dropout_type == "group") config$groups else n_batches
  current_mid   <- runif(dropout_dim, mid_range[1], mid_range[2])
  current_shape <- runif(dropout_dim, shape_range[1], shape_range[2])
  
  params <- setParams(
    base_params,
    nGenes = 1000, batchCells = batch_cells, group.prob = config$props, de.prob = 0.3,
    dropout.mid = current_mid, dropout.shape = current_shape, dropout.type = dropout_type
  )
  
  # 1. Simulate
  sce <- splatSimulate(params, method = "groups", verbose = FALSE, batch.rmEffect = rm_batch_effect)
  
  # 2. Normalize (Counts normalized by Counts; TrueCounts by TrueCounts)
  sce <- apply_normalization(sce)
  
  # 3. DimRed
  sce <- runPCA(sce, exprs_values = "logcounts", name = "PCA_noisy", ncomponents = 2)
  sce <- runPCA(sce, exprs_values = "logTrueCounts", name = "PCA_logTrue", ncomponents = 2)
  
  perp <- max(5, min(30, floor((ncol(sce) - 1) / 3)))
  sce <- runTSNE(sce, exprs_values = "logcounts", name = "TSNE_noisy", perplexity = perp)
  sce <- runTSNE(sce, exprs_values = "logTrueCounts", name = "TSNE_logTrue", perplexity = perp)
  
  # 4. Visualize
  sub_str <- sprintf("Batches: %d | Drop: %s | Mid: %s | Shp: %s", 
                     n_batches, dropout_type, 
                     paste(round(current_mid, 2), collapse=","), 
                     paste(round(current_shape, 2), collapse=","))
  
  save_dimred_plot(sce, "PCA", file.path(OUT_DIR, paste0(sim_id, "_PCA.png")), 
                   paste("PCA:", config$name), sub_str)
  save_dimred_plot(sce, "TSNE", file.path(OUT_DIR, paste0(sim_id, "_TSNE.png")), 
                   paste("t-SNE:", config$name), sub_str)
  
  # 5. PREPARE FOR SAVE: Force Dense Matrices & Integer Dropout
  for (nm in assayNames(sce)) {
    mat <- assay(sce, nm)
    
    # Convert sparse to dense if needed
    if (is(mat, "sparseMatrix") || is(mat, "dgCMatrix")) {
      mat <- as.matrix(mat)
    }
    
    # Fix Dropout: Logical -> Integer (0/1)
    if (nm == "Dropout") {
      if (!is.integer(mat)) {
        mat <- matrix(as.integer(mat), nrow=nrow(mat), ncol=ncol(mat), dimnames=dimnames(mat))
      }
    }
    
    assay(sce, nm) <- mat
  }

  # Save Full Object
  saveRDS(sce, file = file.path(OUT_DIR, paste0(sim_id, ".rds")))
  cat(sprintf("   > Saved RDS (Dense Matrices): %s.rds\n", sim_id))
}

cat("\nAll simulations completed successfully.\n")
