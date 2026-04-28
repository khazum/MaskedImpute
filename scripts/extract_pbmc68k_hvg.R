#!/usr/bin/env Rscript
# Extract a label-free HVG subset from the 10x PBMC68k RDS artifact.

suppressPackageStartupMessages(library(Matrix))

args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(flag, default = NULL) {
  idx <- match(flag, args)
  if (is.na(idx) || idx == length(args)) return(default)
  args[[idx + 1]]
}
has_flag <- function(flag) flag %in% args

input <- get_arg("--input", "temp/pbmc68k_data.rds")
annotations <- get_arg("--annotations", "temp/single-cell-3prime-paper/pbmc68k_analysis/68k_pbmc_barcodes_annotation.tsv")
out_dir <- get_arg("--out-dir", "results_real_data/cache/pbmc68k")
n_hvg <- as.integer(get_arg("--n-hvg", "1000"))
max_cells <- as.integer(get_arg("--max-cells", "12000"))
seed <- as.integer(get_arg("--seed", "42"))
sampling <- get_arg("--sampling", "random")
target_sum <- as.numeric(get_arg("--target-sum", "10000"))
min_cells <- as.integer(get_arg("--min-cells", "10"))

if (!file.exists(input)) stop("PBMC input RDS not found: ", input)
if (!file.exists(annotations)) stop("PBMC annotations not found: ", annotations)
if (!sampling %in% c("none", "random", "stratified")) stop("Unsupported sampling: ", sampling)

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
set.seed(seed)

pbmc <- readRDS(input)
mat <- pbmc$all_data[[1]]$hg19$mat
barcodes <- pbmc$all_data[[1]]$hg19$barcodes
genes <- pbmc$all_data[[1]]$hg19$genes
symbols <- pbmc$all_data[[1]]$hg19$gene_symbols
ann <- read.delim(annotations, stringsAsFactors = FALSE, check.names = FALSE)
if (!all(c("barcodes", "celltype") %in% colnames(ann))) {
  stop("Annotation file must contain barcodes and celltype columns")
}
ann_idx <- match(barcodes, ann$barcodes)
keep <- which(!is.na(ann_idx) & nzchar(ann$celltype[ann_idx]))
if (length(keep) < 2) stop("Too few annotated PBMC cells after barcode matching")

if (!is.na(max_cells) && max_cells > 0 && length(keep) > max_cells && sampling != "none") {
  if (sampling == "stratified") {
    labels <- ann$celltype[ann_idx[keep]]
    split_idx <- split(keep, labels)
    per_group <- pmax(1L, floor(max_cells * lengths(split_idx) / length(keep)))
    selected <- unlist(Map(function(idx, n) sample(idx, min(length(idx), n)), split_idx, per_group), use.names = FALSE)
    if (length(selected) < max_cells) {
      rest <- setdiff(keep, selected)
      selected <- c(selected, sample(rest, min(length(rest), max_cells - length(selected))))
    }
    keep <- sort(selected[seq_len(min(length(selected), max_cells))])
  } else {
    keep <- sort(sample(keep, max_cells))
  }
}

counts_all <- mat[keep, , drop = FALSE]
lib <- Matrix::rowSums(counts_all)
valid_cells <- which(lib > 0)
counts_all <- counts_all[valid_cells, , drop = FALSE]
keep <- keep[valid_cells]
lib <- lib[valid_cells]

scale <- target_sum / pmax(lib, 1)
log_all <- Diagonal(x = scale) %*% counts_all
log_all@x <- log2(1 + log_all@x)

expr_cells <- Matrix::colSums(counts_all > 0)
mu <- Matrix::colMeans(log_all)
mu2 <- Matrix::colMeans(log_all ^ 2)
var <- pmax(mu2 - mu ^ 2, 0)
disp <- var / pmax(mu, 1e-8)
valid_genes <- which(expr_cells >= min_cells & is.finite(disp) & mu > 0)
if (length(valid_genes) < n_hvg) {
  warning("Requested ", n_hvg, " HVGs but only ", length(valid_genes), " genes pass filters")
  n_hvg <- length(valid_genes)
}

# Normalize dispersion within expression bins to avoid selecting only high-expression genes.
rank_mu <- rank(mu[valid_genes], ties.method = "first")
bins <- cut(rank_mu, breaks = unique(floor(seq(0, length(rank_mu), length.out = 21))), include.lowest = TRUE, labels = FALSE)
score <- rep(NA_real_, length(valid_genes))
for (b in sort(unique(bins))) {
  idx <- which(bins == b)
  d <- log1p(disp[valid_genes[idx]])
  center <- median(d, na.rm = TRUE)
  spread <- mad(d, center = center, constant = 1, na.rm = TRUE)
  if (!is.finite(spread) || spread <= 0) spread <- sd(d, na.rm = TRUE)
  if (!is.finite(spread) || spread <= 0) spread <- 1
  score[idx] <- (d - center) / spread
}
score[!is.finite(score)] <- -Inf
hvg <- valid_genes[order(score, decreasing = TRUE)[seq_len(n_hvg)]]
counts <- as(counts_all[, hvg, drop = FALSE], "dgTMatrix")
logcounts <- as(log_all[, hvg, drop = FALSE], "dgTMatrix")

invisible(Matrix::writeMM(counts, file.path(out_dir, "counts.mtx")))
invisible(Matrix::writeMM(logcounts, file.path(out_dir, "logcounts.mtx")))
meta <- data.frame(
  cell_id = barcodes[keep],
  label = ann$celltype[ann_idx[keep]],
  batch = sub("^.*-", "gem_group_", barcodes[keep]),
  stringsAsFactors = FALSE
)
write.table(meta, file.path(out_dir, "metadata.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
features <- data.frame(
  gene_id = genes[hvg],
  gene_symbol = symbols[hvg],
  source_index = hvg,
  hvg_score = score[match(hvg, valid_genes)],
  stringsAsFactors = FALSE
)
write.table(features, file.path(out_dir, "features.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
writeLines(c(
  paste0("target_sum\t", target_sum),
  paste0("n_cells\t", nrow(counts)),
  paste0("n_genes\t", ncol(counts)),
  paste0("sampling\t", sampling),
  paste0("seed\t", seed),
  paste0("max_cells\t", max_cells)
), file.path(out_dir, "normalization.tsv"))
cat("Wrote PBMC68k HVG cache to ", out_dir, " with ", nrow(counts), " cells and ", ncol(counts), " genes\n", sep = "")
