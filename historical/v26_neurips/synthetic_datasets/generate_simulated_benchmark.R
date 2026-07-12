#!/usr/bin/env Rscript

# generate_simulated_benchmark.R
# Generates simulated scRNA-seq benchmark datasets (Splatter/Splat) and saves
# only SingleCellExperiment RDS files.

suppressPackageStartupMessages({
  library(splatter)
  library(Matrix)
  library(SingleCellExperiment)
  library(SummarizedExperiment)
})

# -------------------------
# Configuration
# -------------------------
OUTPUT_DIR <- "simulated_data"
OVERWRITE_EXISTING <- FALSE
BASE_SEED <- 1L

NUM_GENES <- 1000L
MIN_EXPRESSED_CELLS <- 3L

TUNE_NUM_CELLS <- 1000L
TEST_NUM_CELLS <- 5000L
SCALE_NUM_CELLS <- as.integer(seq(10000L, 100000L, by = 10000L))

SCALE_SCENARIO_ID <- "groups_balanced_moderate_drop"

SEED_OFFSET_TUNE <- 100000L
SEED_OFFSET_TEST <- 200000L
SEED_OFFSET_SCALE <- 300000L

DEFAULT_SPLAT_PARAMS <- splatter::newSplatParams()
DEFAULT_BATCH_FAC_LOC <- splatter::getParam(DEFAULT_SPLAT_PARAMS, "batch.facLoc")
DEFAULT_BATCH_FAC_SCALE <- splatter::getParam(DEFAULT_SPLAT_PARAMS, "batch.facScale")
DEFAULT_LIB_SCALE <- splatter::getParam(DEFAULT_SPLAT_PARAMS, "lib.scale")

read_env_flag <- function(name, default = FALSE) {
  raw <- trimws(tolower(Sys.getenv(name, if (default) "true" else "false")))
  raw %in% c("1", "true", "yes", "y", "on")
}

read_env_csv <- function(name, default = "") {
  raw <- trimws(Sys.getenv(name, default))
  if (!nzchar(raw)) return(character())
  parts <- trimws(unlist(strsplit(raw, ",")))
  parts[nzchar(parts)]
}

# Optional runtime overrides.
OUTPUT_DIR <- trimws(Sys.getenv("OUTPUT_DIR", OUTPUT_DIR))
OVERWRITE_EXISTING <- read_env_flag("OVERWRITE_EXISTING", OVERWRITE_EXISTING)
GENERATE_SPLITS <- tolower(read_env_csv("GENERATE_SPLITS", "tune,test,scale"))
if (length(GENERATE_SPLITS) == 0) {
  GENERATE_SPLITS <- c("tune", "test", "scale")
}

# -------------------------
# Data processing
# -------------------------
filter_genes_by_min_cells <- function(sce, min_cells) {
  observed_counts <- SummarizedExperiment::assay(sce, "counts")
  keep_gene <- Matrix::rowSums(observed_counts > 0) >= min_cells
  sce[keep_gene, , drop = FALSE]
}

add_truecount_log_normalization <- function(sce, target_sum = 10000L) {
  true_counts <- SummarizedExperiment::assay(sce, "TrueCounts")
  observed_counts <- SummarizedExperiment::assay(sce, "counts")

  # --- 1. Normalize Observed Counts (CP10k) ---
  obs_library_sizes <- Matrix::colSums(observed_counts)
  obs_library_denominator <- ifelse(obs_library_sizes > 0, obs_library_sizes, 1)
  obs_column_scale <- target_sum / obs_library_denominator
  SummarizedExperiment::assay(sce, "logcounts") <- as.matrix(log2(1 + sweep(observed_counts, 2, obs_column_scale, "*")))

  # --- 2. Normalize True Counts (CP10k) ---
  true_library_sizes <- Matrix::colSums(true_counts)
  true_library_denominator <- ifelse(true_library_sizes > 0, true_library_sizes, 1)
  true_column_scale <- target_sum / true_library_denominator
  SummarizedExperiment::assay(sce, "logTrueCounts") <- as.matrix(log2(1 + sweep(true_counts, 2, true_column_scale, "*")))

  # --- 3. Save Metadata ---
  SummarizedExperiment::colData(sce)$libSizeObserved <- obs_library_sizes
  SummarizedExperiment::colData(sce)$libSizeTrue <- true_library_sizes
  SummarizedExperiment::colData(sce)$targetSum <- rep(as.numeric(target_sum), ncol(sce))

  S4Vectors::metadata(sce)$normalization <- list(
    method = "CP10k",
    target_sum = target_sum,
    observed_library_sizes = obs_library_sizes,
    true_library_sizes = true_library_sizes
  )

  sce
}

sanitize_assays_for_export <- function(sce) {
  required_assays <- c("counts", "TrueCounts", "logcounts", "logTrueCounts")
  available <- SummarizedExperiment::assayNames(sce)
  missing <- setdiff(required_assays, available)
  if (length(missing) > 0) {
    stop("Missing required assay(s): ", paste(missing, collapse = ", "), call. = FALSE)
  }

  assays_list <- SummarizedExperiment::assays(sce, withDimnames = TRUE)
  changed <- character()

  for (nm in names(assays_list)) {
    assay_obj <- assays_list[[nm]]
    if (methods::is(assay_obj, "dgeMatrix")) {
      assays_list[[nm]] <- as.matrix(assay_obj)
      changed <- c(changed, nm)
    } else if (methods::is(assay_obj, "lMatrix")) {
      # rds2py cannot coerce logical sparse matrices; store as numeric sparse 0/1.
      assays_list[[nm]] <- methods::as(assay_obj, "dgCMatrix")
      changed <- c(changed, nm)
    } else if (is.matrix(assay_obj) && is.logical(assay_obj)) {
      # rds2py cannot coerce logical dense matrices; store as integer 0/1.
      storage.mode(assay_obj) <- "integer"
      assays_list[[nm]] <- assay_obj
      changed <- c(changed, nm)
    } else if (methods::is(assay_obj, "Matrix") && !methods::is(assay_obj, "dgCMatrix")) {
      # Prefer sparse numeric representation for unsupported Matrix subclasses.
      coerced <- tryCatch(methods::as(assay_obj, "dgCMatrix"), error = function(e) NULL)
      if (is.null(coerced)) {
        assays_list[[nm]] <- as.matrix(assay_obj)
      } else {
        assays_list[[nm]] <- coerced
      }
      changed <- c(changed, nm)
    }
  }

  SummarizedExperiment::assays(sce, withDimnames = TRUE) <- assays_list
  list(sce = sce, changed = unique(changed))
}

ensure_targetsum_metadata <- function(sce, default_target_sum = 10000) {
  changed <- FALSE
  coldata <- SummarizedExperiment::colData(sce)
  md <- S4Vectors::metadata(sce)
  md_norm <- md$normalization
  if (is.null(md_norm) || !is.list(md_norm)) md_norm <- list()

  target_sum <- NA_real_
  if ("targetSum" %in% colnames(coldata)) {
    vals <- as.numeric(coldata$targetSum)
    vals <- vals[is.finite(vals) & vals > 0]
    if (length(vals) > 0) target_sum <- vals[1]
  }
  if ((!is.finite(target_sum) || target_sum <= 0) && !is.null(md_norm$target_sum)) {
    ts <- as.numeric(md_norm$target_sum)[1]
    if (is.finite(ts) && ts > 0) target_sum <- ts
  }
  if (!is.finite(target_sum) || target_sum <= 0) {
    target_sum <- as.numeric(default_target_sum)
  }

  needs_col <- !("targetSum" %in% colnames(coldata)) ||
    any(!is.finite(as.numeric(coldata$targetSum)) | as.numeric(coldata$targetSum) <= 0)
  if (needs_col) {
    SummarizedExperiment::colData(sce)$targetSum <- rep(target_sum, ncol(sce))
    changed <- TRUE
  }

  if (is.null(md$normalization) || is.null(md$normalization$target_sum) ||
      !is.finite(as.numeric(md$normalization$target_sum)[1]) ||
      as.numeric(md$normalization$target_sum)[1] <= 0) {
    md_norm$target_sum <- target_sum
    md$normalization <- md_norm
    S4Vectors::metadata(sce) <- md
    changed <- TRUE
  }

  list(sce = sce, changed = changed, target_sum = target_sum)
}

repair_existing_dataset_if_needed <- function(output_path, label) {
  obj <- readRDS(output_path)
  if (!inherits(obj, "SingleCellExperiment")) {
    warning(sprintf("%s existing file is not a SingleCellExperiment: %s", label, output_path))
    return(invisible(FALSE))
  }

  repaired <- sanitize_assays_for_export(obj)
  ts_fixed <- ensure_targetsum_metadata(repaired$sce)
  updated <- ts_fixed$sce
  changed <- unique(c(repaired$changed, if (isTRUE(ts_fixed$changed)) "targetSum" else character()))
  if (length(changed) > 0) {
    saveRDS(updated, output_path)
    message(
      label, " repaired dataset fields at ", output_path, ": ",
      paste(changed, collapse = ", ")
    )
    return(invisible(TRUE))
  }

  required_assays <- c("counts", "TrueCounts", "logcounts", "logTrueCounts")
  if (setequal(SummarizedExperiment::assayNames(updated), required_assays)) {
    message(
      label, " note: existing dataset has only core assays at ", output_path,
      ". Re-run with OVERWRITE_EXISTING=true to regenerate extra assays."
    )
  }

  invisible(FALSE)
}

# -------------------------
# Scenario catalog
# -------------------------
create_group_scenario <- function(
  id,
  group_prob,
  de_prob,
  dropout_type,
  dropout_mid = 0,
  dropout_shape = 0,
  extra_args_fn = NULL
) {
  list(
    id = id,
    generate = function(n_cells, seed) {
      extra_args <- if (is.null(extra_args_fn)) list() else extra_args_fn(n_cells)
      params <- splatter::setParams(
        DEFAULT_SPLAT_PARAMS,
        seed = seed,
        nGenes = NUM_GENES,
        batchCells = if (is.null(extra_args$batch_cells)) n_cells else extra_args$batch_cells,
        group.prob = group_prob,
        de.prob = de_prob,
        batch.facLoc = if (is.null(extra_args$batch_facLoc)) DEFAULT_BATCH_FAC_LOC else extra_args$batch_facLoc,
        batch.facScale = if (is.null(extra_args$batch_facScale)) DEFAULT_BATCH_FAC_SCALE else extra_args$batch_facScale,
        lib.scale = if (is.null(extra_args$lib_scale)) DEFAULT_LIB_SCALE else extra_args$lib_scale,
        dropout.mid = dropout_mid,
        dropout.shape = dropout_shape
      )
      params <- splatter::setParam(params, "dropout.type", dropout_type)
      splatter::splatSimulate(params = params, method = "groups", verbose = FALSE)
    }
  )
}

create_path_scenario <- function(
  id,
  group_prob,
  de_prob,
  de_facLoc,
  path_from,
  path_nSteps,
  dropout_type,
  dropout_mid,
  dropout_shape
) {
  list(
    id = id,
    generate = function(n_cells, seed) {
      params <- splatter::setParams(
        DEFAULT_SPLAT_PARAMS,
        seed = seed,
        nGenes = NUM_GENES,
        batchCells = n_cells,
        group.prob = group_prob,
        de.prob = de_prob,
        de.facLoc = de_facLoc,
        path.from = path_from,
        path.nSteps = path_nSteps,
        dropout.mid = dropout_mid,
        dropout.shape = dropout_shape
      )
      params <- splatter::setParam(params, "dropout.type", dropout_type)
      splatter::splatSimulate(params = params, method = "paths", verbose = FALSE)
    }
  )
}

make_group_specific_values <- function(base_value, n_groups, half_range) {
  if (n_groups <= 1L) {
    return(base_value)
  }
  as.numeric(base_value + seq(-half_range, half_range, length.out = n_groups))
}

create_group_dropout_variants <- function(base_id, group_prob, de_prob, dropout_mid, dropout_shape, extra_args_fn = NULL) {
  n_groups <- length(group_prob)
  list(
    create_group_scenario(
      id = base_id,
      group_prob = group_prob,
      de_prob = de_prob,
      dropout_type = "experiment",
      dropout_mid = dropout_mid,
      dropout_shape = dropout_shape,
      extra_args_fn = extra_args_fn
    ),
    create_group_scenario(
      id = paste0(base_id, "_group"),
      group_prob = group_prob,
      de_prob = de_prob,
      dropout_type = "group",
      dropout_mid = make_group_specific_values(dropout_mid, n_groups, half_range = 0.4),
      dropout_shape = make_group_specific_values(dropout_shape, n_groups, half_range = 0.25),
      extra_args_fn = extra_args_fn
    )
  )
}

create_path_dropout_variants <- function(base_id, group_prob, de_prob, de_facLoc, path_from, path_nSteps, dropout_mid, dropout_shape) {
  n_groups <- length(group_prob)
  list(
    create_path_scenario(
      id = base_id,
      group_prob = group_prob,
      de_prob = de_prob,
      de_facLoc = de_facLoc,
      path_from = path_from,
      path_nSteps = path_nSteps,
      dropout_type = "experiment",
      dropout_mid = dropout_mid,
      dropout_shape = dropout_shape
    ),
    create_path_scenario(
      id = paste0(base_id, "_group"),
      group_prob = group_prob,
      de_prob = de_prob,
      de_facLoc = de_facLoc,
      path_from = path_from,
      path_nSteps = path_nSteps,
      dropout_type = "group",
      dropout_mid = make_group_specific_values(dropout_mid, n_groups, half_range = 0.4),
      dropout_shape = make_group_specific_values(dropout_shape, n_groups, half_range = 0.25)
    )
  )
}

build_scenarios <- function() {
  batch_extra_args <- function(n_cells) {
    first_batch <- as.integer(floor(n_cells / 2))
    second_batch <- as.integer(n_cells - first_batch)
    list(
      batch_cells = c(first_batch, second_batch),
      batch_facLoc = 0.10,
      batch_facScale = 0.10,
      lib_scale = 0.40
    )
  }

  group_variant_specs <- list(
    list(base_id = "groups_balanced_moderate_drop", group_prob = rep(1 / 3, 3), de_prob = 0.10, dropout_mid = 3.0, dropout_shape = -1.0),
    list(base_id = "groups_imbalanced_moderate_drop", group_prob = c(0.6, 0.3, 0.1), de_prob = 0.10, dropout_mid = 3.0, dropout_shape = -1.0),
    list(base_id = "groups_rare_high_drop", group_prob = c(0.50, 0.25, 0.20, 0.05), de_prob = 0.10, dropout_mid = 4.0, dropout_shape = -1.0),
    list(base_id = "batch_effects_moderate_drop", group_prob = rep(1 / 3, 3), de_prob = 0.10, dropout_mid = 3.0, dropout_shape = -1.0, extra_args_fn = batch_extra_args)
  )

  path_variant_specs <- list(
    list(base_id = "paths_linear_moderate_drop", group_prob = rep(0.25, 4), de_prob = 0.20, de_facLoc = 0.20, path_from = c(0, 1, 2, 3), path_nSteps = 50, dropout_mid = 3.0, dropout_shape = -1.0),
    list(base_id = "paths_branching_moderate_drop", group_prob = rep(0.25, 4), de_prob = 0.20, de_facLoc = 0.20, path_from = c(0, 1, 1, 3), path_nSteps = 50, dropout_mid = 3.0, dropout_shape = -1.0)
  )

  group_variants <- unlist(lapply(group_variant_specs, function(spec) do.call(create_group_dropout_variants, spec)), recursive = FALSE)
  path_variants <- unlist(lapply(path_variant_specs, function(spec) do.call(create_path_dropout_variants, spec)), recursive = FALSE)

  c(
    list(create_group_scenario(id = "groups_balanced_nodrop", group_prob = rep(1 / 3, 3), de_prob = 0.10, dropout_type = "none")),
    group_variants,
    path_variants
  )
}

# -------------------------
# Generation
# -------------------------
generate_dataset_if_needed <- function(dataset_dir, label, scenario, n_cells, seed) {
  output_path <- file.path(dataset_dir, "sce.rds")
  if (file.exists(output_path) && !OVERWRITE_EXISTING) {
    repair_existing_dataset_if_needed(output_path, label)
    return(invisible(FALSE))
  }
  if (dir.exists(dataset_dir) && OVERWRITE_EXISTING) {
    unlink(dataset_dir, recursive = TRUE, force = TRUE)
  }

  message(label, " ", scenario$id, " n_cells=", n_cells, " seed=", seed)
  sce <- scenario$generate(n_cells = n_cells, seed = seed)
  sce <- filter_genes_by_min_cells(sce, MIN_EXPRESSED_CELLS)
  sce <- add_truecount_log_normalization(sce)
  sanitized <- sanitize_assays_for_export(sce)
  sce <- sanitized$sce
  sce <- ensure_targetsum_metadata(sce)$sce

  if (!dir.exists(dataset_dir)) {
    dir.create(dataset_dir, recursive = TRUE, showWarnings = FALSE)
  }
  saveRDS(sce, output_path)

  invisible(TRUE)
}

run_generation <- function() {
  if (!dir.exists(OUTPUT_DIR)) {
    dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
  }

  scenarios <- build_scenarios()

  split_specs <- list()
  if ("tune" %in% GENERATE_SPLITS) {
    split_specs[[length(split_specs) + 1]] <- list(
      name = "tune", label = "[tune]", n_cells = TUNE_NUM_CELLS, seed_offset = SEED_OFFSET_TUNE
    )
  }
  if ("test" %in% GENERATE_SPLITS) {
    split_specs[[length(split_specs) + 1]] <- list(
      name = "test", label = "[test]", n_cells = TEST_NUM_CELLS, seed_offset = SEED_OFFSET_TEST
    )
  }

  for (split in split_specs) {
    for (scenario_index in seq_along(scenarios)) {
      scenario <- scenarios[[scenario_index]]
      dataset_dir <- file.path(OUTPUT_DIR, split$name, scenario$id)
      seed <- as.integer(BASE_SEED + split$seed_offset + scenario_index * 100L)
      generate_dataset_if_needed(dataset_dir, split$label, scenario, split$n_cells, seed)
    }
  }

  if ("scale" %in% GENERATE_SPLITS) {
    scale_matches <- Filter(function(s) identical(s$id, SCALE_SCENARIO_ID), scenarios)
    if (length(scale_matches) != 1L) {
      stop("Missing or non-unique scale scenario: ", SCALE_SCENARIO_ID, call. = FALSE)
    }
    scale_scenario <- scale_matches[[1]]

    for (n_cells in SCALE_NUM_CELLS) {
      dataset_dir <- file.path(OUTPUT_DIR, "scale", scale_scenario$id, paste0("n", n_cells))
      seed <- as.integer(BASE_SEED + SEED_OFFSET_SCALE + as.integer(n_cells))
      generate_dataset_if_needed(dataset_dir, "[scale]", scale_scenario, n_cells, seed)
    }
  }

  invisible(TRUE)
}

run_generation()
