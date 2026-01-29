#!/usr/bin/env Rscript

# Description:
# This script processes a directory of pre-normalized SingleCellExperiment (SCE) objects.
# For each SCE, it selects the top N marker genes that best distinguish cell types based on
# existing labels in colData. The resulting subsetted SCE object is saved to an output directory.

# --- Load required libraries ---
suppressPackageStartupMessages(library(argparse))
suppressPackageStartupMessages(library(scran))
suppressPackageStartupMessages(library(SingleCellExperiment))

# --- Define Command-Line Arguments ---
parser <- ArgumentParser(description = "Find marker genes in pre-normalized SingleCellExperiment objects.")
parser$add_argument("-i", "--input_dir", type = "character", required = TRUE,
                    help = "Path to the directory containing input RDS files.")
parser$add_argument("-o", "--output_dir", type = "character", required = TRUE,
                    help = "Path to the directory where output RDS files will be saved.")
parser$add_argument("-n", "--n_genes", type = "integer", default = 1000,
                    help = "Number of top marker genes to select [default: %(default)s].")
args <- parser$parse_args()

# --- Main Processing Logic ---

# 1. Validate inputs and find RDS files
if (!dir.exists(args$input_dir)) {
  stop("Input directory does not exist: ", args$input_dir)
}
rds_files <- list.files(path = args$input_dir, pattern = "\\.rds$", full.names = TRUE, ignore.case = TRUE)
if (length(rds_files) == 0) {
  stop("No .rds files found in the specified input directory.")
}

# 2. Create output directory
if (!dir.exists(args$output_dir)) {
  message("Output directory does not exist. Creating it now: ", args$output_dir)
  dir.create(args$output_dir, recursive = TRUE)
}

# 3. Loop through each file
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

    possible_label_cols <- c("cell_type1", "cell_type", "label", "labels", "Group")
    label_col <- NULL
    for (col in possible_label_cols) {
        if (col %in% names(colData(sce))) {
            label_col <- col
            message(paste("  -> Found cell labels in column:", label_col))
            break
        }
    }
    if (is.null(label_col)) {
        warning(paste("Skipping", base_name, " - could not find a valid cell label column."))
        next
    }
    
    n_top_genes <- args$n_genes

    message(paste("  -> Running findMarkers to identify top", n_top_genes, "genes..."))
    # We run findMarkers normally. The fix is applied *after* this step.
    marker_results <- findMarkers(sce, groups = colData(sce)[[label_col]])

    message("  -> Creating a global ranking of all markers...")
    
    # --- THE DEFINITIVE FIX ---
    # We now loop through the results. For each dataframe, we select ONLY the common
    # summary columns before adding gene/cell_type info. This guarantees that every
    # dataframe passed to rbind() has the exact same structure.
    list_of_dfs <- lapply(names(marker_results), function(cell_type) {
      df <- marker_results[[cell_type]]
      
      # Select only the columns that are guaranteed to be present and consistent
      df_subset <- df[, c("summary.logFC", "p.value", "FDR")]
      
      df_subset$gene <- rownames(df)
      df_subset$cell_type_marker <- cell_type
      return(df_subset)
    })
    
    # Now, rbind will work because every element in list_of_dfs has the same columns.
    combined_markers <- do.call(rbind, list_of_dfs)
    
    upregulated_markers <- combined_markers[combined_markers$summary.logFC > 0 & combined_markers$FDR < 0.05, ]
    sorted_markers <- upregulated_markers[order(upregulated_markers$summary.logFC, decreasing = TRUE), ]
    
    if (nrow(sorted_markers) == 0) {
        warning(paste("Skipping", base_name, "- no significant upregulated markers found."))
        next
    }
    top_genes_final <- unique(sorted_markers$gene)[1:n_top_genes]
    top_genes_final <- top_genes_final[!is.na(top_genes_final)] 
    print(length(top_genes_final))
    sce_subset <- sce[top_genes_final, ]
    message(paste("  -> Subset object to", length(top_genes_final), "unique marker genes using global ranking."))

    output_filename <- paste0(base_name, "_top", args$n_genes, "markers.rds")
    output_path <- file.path(args$output_dir, output_filename)

    saveRDS(sce_subset, file = output_path)
    message(paste("  -> Successfully saved subsetted object to:", output_path))

  }, error = function(e) {
    message(paste("\n[ERROR] Failed to process", base_name, ":", e$message, "\n"))
  })
}

message("\n--- All processing complete. ---")
