#!/usr/bin/env Rscript
# run_clustering.R
# Usage: Rscript run_clustering.R <input_rds_file_or_dir> <output_dir> [ncores] [n_repeats] [methods]
#
# Methods: baseline, saver, ccimpute, experiment
# For each dataset/method, compute clustering metrics on logcounts
# using shared Python clustering (PCA + k-means) for fair comparison:
#   ASW, ARI, NMI, Purity (PS)

stopf <- function(fmt, ...) stop(sprintf(fmt, ...), call. = FALSE)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stopf("Usage: Rscript run_clustering.R <input_rds_file_or_dir> <output_dir> [ncores] [n_repeats] [methods]")
}

input_path <- args[1]
output_dir <- args[2]
ncores <- if (length(args) >= 3) as.integer(args[3]) else parallel::detectCores()
if (is.na(ncores) || ncores < 1) ncores <- 1
n_repeats <- if (length(args) >= 4) as.integer(args[4]) else 5L
if (is.na(n_repeats) || n_repeats < 1) n_repeats <- 1L
methods_arg <- if (length(args) >= 5) args[5] else "all"

if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

require_pkg <- function(pkg) {
  if (requireNamespace(pkg, quietly = TRUE)) return(invisible(TRUE))
  stopf("Missing required package '%s'.", pkg)
}

load_pkg <- function(pkg) {
  suppressPackageStartupMessages(library(pkg, character.only = TRUE))
}

core_pkgs <- c("SingleCellExperiment", "Matrix", "parallel", "reticulate")
invisible(lapply(core_pkgs, require_pkg))
invisible(lapply(core_pkgs, load_pkg))

has_cluster <- requireNamespace("cluster", quietly = TRUE)

py_clust <- NULL
py_exp <- NULL

init_python_clustering <- function() {
  if (!is.null(py_clust)) return(invisible(py_clust))
  py_bin <- Sys.getenv("MASKEDIMPUTE_PYTHON")
  if (nzchar(py_bin)) {
    reticulate::use_python(py_bin, required = TRUE)
  }
  proj_root <- normalizePath(getwd())
  reticulate::py_run_string(sprintf("import sys; sys.path.insert(0, r'%s')", proj_root))
  if (!reticulate::py_module_available("numpy")) {
    stopf("Python env missing numpy. Set MASKEDIMPUTE_PYTHON to a Python with numpy.")
  }
  py_clust <<- tryCatch(
    reticulate::import("clustering_eval", delay_load = FALSE),
    error = function(e) stopf("Failed to import clustering_eval via reticulate: %s", conditionMessage(e))
  )
  invisible(py_clust)
}

get_python_experiment <- function() {
  init_python_clustering()
  if (!is.null(py_exp)) return(py_exp)
  py_exp <<- tryCatch(
    reticulate::import("experiment", delay_load = FALSE, convert = FALSE),
    error = function(e) stopf("Failed to import experiment.py via reticulate: %s", conditionMessage(e))
  )
  py_exp
}

parse_methods <- function(raw) {
  if (is.null(raw) || !nzchar(raw) || tolower(raw) == "all") {
    return(c("baseline", "saver", "ccimpute", "experiment"))
  }
  methods <- tolower(unlist(strsplit(raw, ",")))
  methods <- methods[nzchar(methods)]
  allowed <- c("baseline", "saver", "ccimpute", "experiment")
  unknown <- setdiff(methods, allowed)
  if (length(unknown) > 0) {
    stopf("Unknown methods: %s. Allowed: %s or 'all'.",
          paste(unknown, collapse = ", "), paste(allowed, collapse = ", "))
  }
  unique(methods)
}

collect_rds_files <- function(path) {
  if (file.exists(path) && !dir.exists(path)) return(normalizePath(path))
  if (dir.exists(path)) {
    files <- list.files(path, pattern = "\\.rds$", recursive = TRUE, full.names = TRUE)
    return(sort(files))
  }
  stopf("Input path not found: %s", path)
}

label_keys <- c("cell_type1", "labels", "Group", "label")

extract_labels <- function(sce) {
  cd <- colData(sce)
  for (key in label_keys) {
    if (key %in% colnames(cd)) {
      lab <- as.vector(cd[[key]])
      return(list(labels = lab, source = key))
    }
  }
  stopf("No label column found. Tried: %s", paste(label_keys, collapse = ", "))
}

comb2 <- function(x) x * (x - 1) / 2

adjusted_rand_index <- function(true_labels, pred_labels) {
  tab <- table(true_labels, pred_labels)
  n <- sum(tab)
  if (n < 2) return(NA_real_)
  sum_comb <- sum(comb2(tab))
  a <- rowSums(tab)
  b <- colSums(tab)
  sum_a <- sum(comb2(a))
  sum_b <- sum(comb2(b))
  expected <- (sum_a * sum_b) / comb2(n)
  max_index <- 0.5 * (sum_a + sum_b) - expected
  if (max_index == 0) return(0)
  (sum_comb - expected) / max_index
}

normalized_mutual_info <- function(true_labels, pred_labels) {
  tab <- table(true_labels, pred_labels)
  n <- sum(tab)
  if (n == 0) return(NA_real_)
  p_ij <- tab / n
  p_i <- rowSums(p_ij)
  p_j <- colSums(p_ij)
  denom <- outer(p_i, p_j, "*")
  nz <- p_ij > 0 & denom > 0
  I <- sum(p_ij[nz] * log(p_ij[nz] / denom[nz]))
  H_i <- -sum(p_i[p_i > 0] * log(p_i[p_i > 0]))
  H_j <- -sum(p_j[p_j > 0] * log(p_j[p_j > 0]))
  if ((H_i + H_j) == 0) return(1)
  (2 * I) / (H_i + H_j)
}

purity_score <- function(true_labels, pred_labels) {
  tab <- table(true_labels, pred_labels)
  if (length(tab) == 0) return(NA_real_)
  sum(apply(tab, 2, max)) / sum(tab)
}

compute_asw <- function(emb, clusters) {
  if (!has_cluster) return(NA_real_)
  k <- length(unique(clusters))
  n <- nrow(emb)
  if (k < 2 || k >= n) return(NA_real_)
  d <- stats::dist(emb)
  sil <- cluster::silhouette(clusters, d)
  mean(sil[, "sil_width"])
}

evaluate_clustering <- function(log_imp, labels) {
  clust <- init_python_clustering()
  lab_vec <- if (is.factor(labels)) as.character(labels) else as.vector(labels)
  if (!reticulate::is_py_object(log_imp)) {
    log_imp <- as.matrix(log_imp)
  }
  res <- tryCatch(
    clust$evaluate_clustering(log_imp, lab_vec),
    error = function(e) stopf("Python clustering failed: %s", conditionMessage(e))
  )
  data.frame(
    ASW = as.numeric(res$ASW),
    ARI = as.numeric(res$ARI),
    NMI = as.numeric(res$NMI),
    PS  = as.numeric(res$PS),
    stringsAsFactors = FALSE
  )
}

# Normalization helpers (copied from run_imputation.R)
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

normalize_counts_to_logcounts <- function(counts, size_factor) {
  counts <- as.matrix(counts)
  if (ncol(counts) != length(size_factor)) {
    stopf("Size factor length (%d) does not match number of cells (%d).",
          length(size_factor), ncol(counts))
  }
  denom <- ifelse(is.finite(size_factor) & size_factor > 0, size_factor, 1)
  log2(1 + t(t(counts) / denom))
}

summarize_repeats <- function(metrics_list, runtimes, err_msg) {
  metric_cols <- c("ASW", "ARI", "NMI", "PS")
  if (length(metrics_list) > 0) {
    metrics_df <- do.call(rbind, metrics_list)
    means <- sapply(metric_cols, function(cn) mean(metrics_df[[cn]], na.rm = TRUE))
    sds <- sapply(metric_cols, function(cn) if (nrow(metrics_df) > 1) stats::sd(metrics_df[[cn]], na.rm = TRUE) else 0)
    runtime_mean <- mean(runtimes)
    runtime_sd <- if (length(runtimes) > 1) stats::sd(runtimes) else 0
    data.frame(
      ASW = means["ASW"],
      ASW_std = sds["ASW"],
      ARI = means["ARI"],
      ARI_std = sds["ARI"],
      NMI = means["NMI"],
      NMI_std = sds["NMI"],
      PS = means["PS"],
      PS_std = sds["PS"],
      runtime_sec = runtime_mean,
      runtime_sec_std = runtime_sd,
      n_repeats = length(runtimes),
      error = err_msg,
      stringsAsFactors = FALSE
    )
  } else {
    data.frame(
      ASW = NA_real_, ASW_std = NA_real_,
      ARI = NA_real_, ARI_std = NA_real_,
      NMI = NA_real_, NMI_std = NA_real_,
      PS = NA_real_, PS_std = NA_real_,
      runtime_sec = NA_real_,
      runtime_sec_std = NA_real_,
      n_repeats = 0L,
      error = err_msg,
      stringsAsFactors = FALSE
    )
  }
}

methods <- parse_methods(methods_arg)
files <- collect_rds_files(input_path)
if (length(files) == 0) stopf("No .rds files found.")

all_results <- list()

for (path in files) {
  dataset_name <- tools::file_path_sans_ext(basename(path))
  cat(sprintf("\n=== %s ===\n", dataset_name))
  sce <- readRDS(path)
  if (!methods::is(sce, "SingleCellExperiment")) {
    cat("  [ERROR] Unsupported RDS object; skipping.\n")
    next
  }

  if (!"logcounts" %in% assayNames(sce)) {
    cat("  [ERROR] Missing logcounts assay; skipping.\n")
    next
  }

  logcounts <- as.matrix(assay(sce, "logcounts"))
  counts <- NULL
  if ("counts" %in% assayNames(sce)) counts <- assay(sce, "counts")
  lab_info <- extract_labels(sce)
  labels <- lab_info$labels
  n_cells <- ncol(logcounts)
  n_genes <- nrow(logcounts)

  norm_info <- tryCatch(get_normalization_info(sce), error = function(e) NULL)
  size_factor_true <- if (!is.null(norm_info)) norm_info$size_factor else NULL

  if ("baseline" %in% methods) {
    cat(sprintf("Running baseline x%d...\n", n_repeats))
    baseline_metrics <- list()
    baseline_runtimes <- numeric()
    baseline_err <- NA_character_
    for (i in seq_len(n_repeats)) {
      t0 <- proc.time()
      res <- tryCatch({
        df <- evaluate_clustering(t(logcounts), labels)
        df
      }, error = function(e) {
        baseline_err <<- conditionMessage(e)
        cat(sprintf("ERROR [%s/baseline]: %s\n", dataset_name, baseline_err))
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
      summarize_repeats(baseline_metrics, baseline_runtimes, baseline_err),
      n_cells = n_cells,
      n_genes = n_genes,
      label_source = lab_info$source,
      stringsAsFactors = FALSE
    )
    all_results[[length(all_results) + 1]] <- baseline_row
  }

  if ("saver" %in% methods) {
    cat(sprintf("Running saver x%d...\n", n_repeats))
    require_pkg("SAVER")
    if (is.null(counts) || is.null(size_factor_true)) {
      cat("  [saver] Missing counts or normalization info; skipping.\n")
    } else {
      counts_mat <- Matrix::Matrix(counts, sparse = TRUE)
      lib_obs <- Matrix::colSums(counts_mat)
      nonzero_cells <- lib_obs > 0
      if (sum(!nonzero_cells) > 0) {
        cat(sprintf("  [saver] Dropping %d zero-expression cells.\n", sum(!nonzero_cells)))
      }
      saver_cells <- nonzero_cells
      counts_saver <- Matrix::Matrix(counts_mat[, saver_cells, drop = FALSE], sparse = TRUE)
      if (ncol(counts_saver) < 2 || nrow(counts_saver) < 1) {
        saver_err <- "Insufficient cells or genes after filtering."
        cat(sprintf("ERROR [%s/saver]: %s\n", dataset_name, saver_err))
        saver_metrics <- list()
        saver_runtimes <- numeric()
      } else {
        mean_obs <- mean(lib_obs[saver_cells])
        size_factor_obs <- lib_obs[saver_cells] / mean_obs
        saver_metrics <- list()
        saver_runtimes <- numeric()
        saver_err <- NA_character_
        for (i in seq_len(n_repeats)) {
          t0 <- proc.time()
          res <- tryCatch({
            saver_imp <- SAVER::saver(counts_saver, ncores = 1)
            saver_est <- as.matrix(saver_imp$estimate)
            saver_counts <- sweep(saver_est, 2, size_factor_obs, "*")
            log_imp <- logcounts
            log_imp[, saver_cells] <- normalize_counts_to_logcounts(
              saver_counts,
              size_factor_true[saver_cells]
            )
            df <- evaluate_clustering(t(log_imp), labels)
            df
          }, error = function(e) {
            saver_err <<- conditionMessage(e)
            cat(sprintf("ERROR [%s/saver]: %s\n", dataset_name, saver_err))
            NULL
          })
          elapsed <- (proc.time() - t0)["elapsed"]
          if (is.null(res)) break
          saver_metrics[[length(saver_metrics) + 1]] <- res
          saver_runtimes <- c(saver_runtimes, elapsed)
        }
      }
      saver_row <- data.frame(
        dataset = dataset_name,
        method = "saver",
        summarize_repeats(saver_metrics, saver_runtimes, saver_err),
        n_cells = n_cells,
        n_genes = n_genes,
        label_source = lab_info$source,
        stringsAsFactors = FALSE
      )
      all_results[[length(all_results) + 1]] <- saver_row
    }
  }

  if ("ccimpute" %in% methods) {
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
        df <- evaluate_clustering(t(as.matrix(log_imp)), labels)
        df
      }, error = function(e) {
        cc_err <<- conditionMessage(e)
        cat(sprintf("ERROR [%s/ccimpute]: %s\n", dataset_name, cc_err))
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
      summarize_repeats(cc_metrics, cc_runtimes, cc_err),
      n_cells = n_cells,
      n_genes = n_genes,
      label_source = lab_info$source,
      stringsAsFactors = FALSE
    )
    all_results[[length(all_results) + 1]] <- cc_row
  }

  if ("experiment" %in% methods) {
    cat(sprintf("Running experiment x%d...\n", n_repeats))
    exp_mod <- get_python_experiment()
    exp_metrics <- list()
    exp_runtimes <- numeric()
    exp_err <- NA_character_
    for (i in seq_len(n_repeats)) {
      t0 <- proc.time()
      res <- tryCatch({
        seed_val <- 42L + (i - 1L)
        log_imp <- exp_mod$run_experiment_imputation(t(logcounts), seed = seed_val)
        df <- evaluate_clustering(log_imp, labels)
        df
      }, error = function(e) {
        exp_err <<- conditionMessage(e)
        cat(sprintf("ERROR [%s/experiment]: %s\n", dataset_name, exp_err))
        NULL
      })
      elapsed <- (proc.time() - t0)["elapsed"]
      if (is.null(res)) break
      exp_metrics[[length(exp_metrics) + 1]] <- res
      exp_runtimes <- c(exp_runtimes, elapsed)
    }
    exp_row <- data.frame(
      dataset = dataset_name,
      method = "experiment",
      summarize_repeats(exp_metrics, exp_runtimes, exp_err),
      n_cells = n_cells,
      n_genes = n_genes,
      label_source = lab_info$source,
      stringsAsFactors = FALSE
    )
    all_results[[length(all_results) + 1]] <- exp_row
  }
}

if (length(all_results) == 0) stopf("No datasets processed.")

results_df <- do.call(rbind, all_results)

write_method_table <- function(df, method) {
  out_path <- file.path(output_dir, paste0(method, "_clustering_table.tsv"))
  utils::write.table(
    df[df$method == method, ],
    out_path,
    sep = "\t",
    quote = FALSE,
    row.names = FALSE
  )
  cat(sprintf("Wrote %s\n", out_path))
}

for (m in methods) {
  if (any(results_df$method == m)) write_method_table(results_df, m)
}

cat("\nDone.\n")
