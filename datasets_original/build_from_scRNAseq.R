#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)

if (length(args) >= 1L && args[[1]] %in% c("-h", "--help")) {
    cat(
        "Usage:\n",
        "  Rscript build_from_scRNAseq.R [targets...]\n\n",
        "Defaults:\n",
        "  output_dir: current directory\n",
        "  targets:    all\n\n",
        "Targets:\n",
        "  all, baron, campbell, chen, macosko, manno, shekhar, zeisel\n",
        sep = ""
    )
    quit(save = "no", status = 0)
}

suppressPackageStartupMessages({
    library(scRNAseq)
    library(SingleCellExperiment)
    library(S4Vectors)
    library(scuttle)
    library(Matrix)
})

output_dir <- "."
targets <- if (length(args) >= 1L) args else "all"
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

message("Writing legacy-style objects to: ", normalizePath(output_dir, mustWork = FALSE))

split_tokens <- function(x, split_pattern, expected_tokens) {
    pieces <- strsplit(x, split_pattern)
    if (any(lengths(pieces) != expected_tokens)) {
        stop("Could not parse all identifiers with pattern: ", split_pattern)
    }
    matrix(unlist(pieces, use.names = FALSE), ncol = expected_tokens, byrow = TRUE)
}

merge_union_features <- function(mats) {
    all_genes <- sort(unique(unlist(lapply(mats, rownames), use.names = FALSE)))
    filled <- lapply(mats, function(mat) {
        out <- Matrix(0, nrow = length(all_genes), ncol = ncol(mat), sparse = TRUE,
                      dimnames = list(all_genes, colnames(mat)))
        out[match(rownames(mat), all_genes), ] <- mat
        out
    })
    do.call(cbind, filled)
}

median_libsize_logcounts <- function(count_mat) {
    lib_sizes <- Matrix::colSums(count_mat)
    positive <- lib_sizes[lib_sizes > 0]
    med_lib <- if (length(positive)) stats::median(positive) else 1
    sf <- lib_sizes / med_lib
    sf[sf == 0] <- 1
    scuttle::normalizeCounts(
        count_mat,
        size.factors = sf,
        center.size.factors = FALSE,
        log = TRUE,
        pseudo.count = 1
    )
}

add_legacy_qc <- function(sce) {
    count_mat <- counts(sce)
    total_counts <- Matrix::colSums(count_mat)
    total_features <- Matrix::colSums(count_mat > 0)

    colData(sce)$total_counts <- as.numeric(total_counts)
    colData(sce)$total_features <- as.numeric(total_features)
    colData(sce)$log10_total_counts <- log10(total_counts + 1)

    rowData(sce)$mean_counts <- as.numeric(Matrix::rowMeans(count_mat))
    rowData(sce)$n_cells_counts <- as.numeric(Matrix::rowSums(count_mat > 0))

    ercc_rows <- as.logical(rowData(sce)$is_ERCC)
    ercc_rows[is.na(ercc_rows)] <- FALSE
    if (any(ercc_rows)) {
        ercc_counts <- Matrix::colSums(count_mat[ercc_rows, , drop = FALSE])
        colData(sce)$total_counts_ERCC <- as.numeric(ercc_counts)
        colData(sce)$pct_counts_ERCC <- ifelse(total_counts > 0, 100 * ercc_counts / total_counts, NA_real_)
    }

    sce
}

build_from_counts <- function(count_mat, ann_df) {
    ann_df <- DataFrame(ann_df)
    stopifnot(identical(colnames(count_mat), rownames(ann_df)))

    sce <- SingleCellExperiment(assays = list(counts = count_mat), colData = ann_df)
    logcounts(sce) <- median_libsize_logcounts(counts(sce))
    rowData(sce)$feature_symbol <- rownames(sce)
    rowData(sce)$is_ERCC <- grepl("^ERCC-", rownames(sce))
    sce <- sce[!duplicated(rowData(sce)$feature_symbol), ]
    add_legacy_qc(sce)
}

save_object <- function(sce, filename) {
    out <- file.path(output_dir, filename)
    saveRDS(sce, out)
    message(sprintf("Saved %s (%d features x %d cells)", out, nrow(sce), ncol(sce)))
}

build_baron <- function() {
    human <- BaronPancreasData("human")
    human_levels <- unique(colData(human)$donor)
    h_ann <- data.frame(
        human = as.integer(match(colData(human)$donor, human_levels)),
        cell_type1 = as.character(colData(human)$label),
        row.names = colnames(human),
        stringsAsFactors = FALSE,
        check.names = FALSE
    )
    save_object(build_from_counts(counts(human), h_ann), "baron-human.rds")

    mouse <- BaronPancreasData("mouse")
    mouse_levels <- unique(colData(mouse)$strain)
    m_ann <- data.frame(
        mouse = as.integer(match(colData(mouse)$strain, mouse_levels)),
        cell_type1 = as.character(colData(mouse)$label),
        row.names = colnames(mouse),
        stringsAsFactors = FALSE,
        check.names = FALSE
    )
    save_object(build_from_counts(counts(mouse), m_ann), "baron-mouse.rds")
}

build_campbell <- function() {
    src <- CampbellBrainData()
    ann <- as.data.frame(colData(src), stringsAsFactors = FALSE)
    if ("clust_neurons" %in% colnames(ann)) {
        colnames(ann)[colnames(ann) == "clust_neurons"] <- "cell_type1"
    } else {
        stop("Could not find 'clust_neurons' in Campbell metadata")
    }
    save_object(build_from_counts(counts(src), ann), "campbell.rds")
}

build_chen <- function() {
    src <- ChenBrainData()
    ann <- data.frame(
        cell_type1 = as.character(colData(src)$SVM_clusterID),
        row.names = colnames(src),
        stringsAsFactors = FALSE,
        check.names = FALSE
    )
    save_object(build_from_counts(counts(src), ann), "chen.rds")
}

build_macosko <- function() {
    src <- MacoskoRetinaData()
    keep <- !is.na(colData(src)$cluster)
    src <- src[, keep]

    clust <- as.integer(as.character(colData(src)$cluster))
    cell_type1 <- rep("rods", length(clust))
    cell_type1[clust == 1] <- "horizontal"
    cell_type1[clust == 2] <- "ganglion"
    cell_type1[clust %in% 3:23] <- "amacrine"
    cell_type1[clust == 25] <- "cones"
    cell_type1[clust %in% 26:33] <- "bipolar"
    cell_type1[clust == 34] <- "muller"
    cell_type1[clust == 35] <- "astrocytes"
    cell_type1[clust == 36] <- "fibroblasts"
    cell_type1[clust == 37] <- "vascular_endothelium"
    cell_type1[clust == 38] <- "pericytes"
    cell_type1[clust == 39] <- "microglia"

    ann <- data.frame(
        clust_id = clust,
        cell_type1 = cell_type1,
        row.names = colnames(src),
        stringsAsFactors = FALSE,
        check.names = FALSE
    )
    save_object(build_from_counts(counts(src), ann), "macosko.rds")
}

build_manno <- function() {
    h_es <- LaMannoBrainData("human-es")
    h_emb <- LaMannoBrainData("human-embryo")
    h_ips <- LaMannoBrainData("human-ips")

    h_counts <- merge_union_features(list(counts(h_es), counts(h_emb), counts(h_ips)))
    h_cell_type <- c(
        as.character(colData(h_es)$Cell_type),
        as.character(colData(h_emb)$Cell_type),
        as.character(colData(h_ips)$Cell_type)
    )
    h_age <- c(
        as.character(colData(h_es)$Timepoint),
        as.character(colData(h_emb)$Timepoint),
        as.character(colData(h_ips)$Timepoint)
    )
    h_source <- rep(
        c("ESCs", "ventral midbrain", "iPSCs"),
        times = c(ncol(h_es), ncol(h_emb), ncol(h_ips))
    )
    h_tokens <- split_tokens(colnames(h_counts), "-|_", 3)
    h_ann <- data.frame(
        Species = rep("Homo sapiens", ncol(h_counts)),
        cell_type1 = h_cell_type,
        Source = h_source,
        age = h_age,
        WellID = h_tokens[, 3],
        batch = paste(h_tokens[, 1], h_tokens[, 2]),
        row.names = colnames(h_counts),
        stringsAsFactors = FALSE,
        check.names = FALSE
    )
    save_object(build_from_counts(h_counts, h_ann), "manno_human.rds")

    m_adult <- LaMannoBrainData("mouse-adult")
    m_emb <- LaMannoBrainData("mouse-embryo")

    m_counts <- merge_union_features(list(counts(m_adult), counts(m_emb)))
    m_cell_type <- c(
        as.character(colData(m_adult)$Cell_type),
        as.character(colData(m_emb)$Cell_type)
    )
    m_age <- c(rep("adult", ncol(m_adult)), as.character(colData(m_emb)$Timepoint))
    m_source <- rep(
        c("substantia nigra-ventral tegmental area", "ventral midbrain"),
        times = c(ncol(m_adult), ncol(m_emb))
    )
    m_tokens <- split_tokens(colnames(m_counts), "-|_", 2)
    m_ann <- data.frame(
        Species = rep("Mus musclus", ncol(m_counts)),
        cell_type1 = m_cell_type,
        Source = m_source,
        age = m_age,
        WellID = m_tokens[, 2],
        batch = m_tokens[, 1],
        row.names = colnames(m_counts),
        stringsAsFactors = FALSE,
        check.names = FALSE
    )
    save_object(build_from_counts(m_counts, m_ann), "manno_mouse.rds")
}

build_shekhar <- function() {
    src <- ShekharRetinaData()
    mat <- counts(src)

    mt_genes <- grep("mt-", rownames(mat), value = TRUE)
    cells_keep <- Matrix::colSums(mat[mt_genes, , drop = FALSE]) / Matrix::colSums(mat) < 0.1
    mat <- mat[, cells_keep, drop = FALSE]
    mat <- mat[, Matrix::colSums(mat > 0) > 500, drop = FALSE]
    mat <- mat[Matrix::rowSums(mat > 0) > 30 & Matrix::rowSums(mat) > 60, , drop = FALSE]

    ann <- data.frame(
        cell_type2 = as.character(colData(src)[colnames(mat), "CLUSTER"]),
        row.names = colnames(mat),
        stringsAsFactors = FALSE,
        check.names = FALSE
    )

    ann$clust_id <- NA_integer_
    ann$clust_id[ann$cell_type2 == "BC1A"] <- 7L
    ann$clust_id[ann$cell_type2 == "BC1B"] <- 9L
    ann$clust_id[ann$cell_type2 == "BC2"] <- 10L
    ann$clust_id[ann$cell_type2 == "BC3A"] <- 12L
    ann$clust_id[ann$cell_type2 == "BC3B"] <- 8L
    ann$clust_id[ann$cell_type2 == "BC4"] <- 14L
    ann$clust_id[ann$cell_type2 == "BC5A (Cone Bipolar cell 5A)"] <- 3L
    ann$clust_id[ann$cell_type2 == "BC5B"] <- 13L
    ann$clust_id[ann$cell_type2 == "BC5C"] <- 6L
    ann$clust_id[ann$cell_type2 == "BC5D"] <- 11L
    ann$clust_id[ann$cell_type2 == "BC6"] <- 5L
    ann$clust_id[ann$cell_type2 == "BC7 (Cone Bipolar cell 7)"] <- 4L
    ann$clust_id[ann$cell_type2 == "BC8/9 (mixture of BC8 and BC9)"] <- 15L
    ann$clust_id[ann$cell_type2 == "RBC (Rod Bipolar cell)"] <- 1L
    ann$clust_id[ann$cell_type2 == "MG (Mueller Glia)"] <- 2L
    ann$clust_id[ann$cell_type2 == "AC (Amacrine cell)"] <- 16L
    ann$clust_id[ann$cell_type2 == "Rod Photoreceptors"] <- 20L
    ann$clust_id[ann$cell_type2 == "Cone Photoreceptors"] <- 22L

    ann$cell_type1 <- "unknown"
    ann$cell_type1[grepl("BC", ann$cell_type2)] <- "bipolar"
    ann$cell_type1[grepl("MG", ann$cell_type2)] <- "muller"
    ann$cell_type1[grepl("AC", ann$cell_type2)] <- "amacrine"
    ann$cell_type1[grepl("Rod Photoreceptors", ann$cell_type2)] <- "rods"
    ann$cell_type1[grepl("Cone Photoreceptors", ann$cell_type2)] <- "cones"

    save_object(build_from_counts(mat, ann), "shekhar.rds")
}

build_zeisel <- function() {
    src <- ZeiselBrainData()
    keep <- rowData(src)$featureType == "endogenous"
    mat <- counts(src)[keep, , drop = FALSE]

    clust_id <- as.character(colData(src)[["group #"]])
    cell_type1 <- clust_id
    cell_type1[clust_id == "1"] <- "interneurons"
    cell_type1[clust_id == "2"] <- "s1pyramidal"
    cell_type1[clust_id == "3"] <- "ca1pyramidal"
    cell_type1[clust_id == "4"] <- "oligodendrocytes"
    cell_type1[clust_id == "5"] <- "microglia"
    cell_type1[clust_id == "6"] <- "endothelial"
    cell_type1[clust_id == "7"] <- "astrocytes"
    cell_type1[clust_id == "8"] <- "ependymal"
    cell_type1[clust_id == "9"] <- "mural"

    ann <- data.frame(
        clust_id = clust_id,
        cell_type1 = cell_type1,
        row.names = colnames(mat),
        stringsAsFactors = FALSE,
        check.names = FALSE
    )
    save_object(build_from_counts(mat, ann), "zeisel.rds")
}

builders <- list(
    baron = build_baron,
    campbell = build_campbell,
    chen = build_chen,
    macosko = build_macosko,
    manno = build_manno,
    shekhar = build_shekhar,
    zeisel = build_zeisel
)

if ("all" %in% targets) {
    selected <- names(builders)
} else {
    selected <- unique(targets)
    unknown <- setdiff(selected, names(builders))
    if (length(unknown)) {
        stop(
            "Unknown target(s): ",
            paste(unknown, collapse = ", "),
            ". Valid targets: ",
            paste(c("all", names(builders)), collapse = ", ")
        )
    }
}

for (nm in selected) {
    message("Building target: ", nm)
    builders[[nm]]()
}

message("Done.")
