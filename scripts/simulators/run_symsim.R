#!/usr/bin/env Rscript

fail <- function(message) {
  stop(message, call. = FALSE)
}

assert_names <- function(value, expected, name) {
  observed <- names(value)
  if (is.null(observed) || !setequal(observed, expected) || length(observed) != length(expected)) {
    fail(sprintf("%s has an invalid schema", name))
  }
}

scalar_integer <- function(value, name, minimum = 0L) {
  if (length(value) != 1L || is.logical(value) || !is.numeric(value) ||
      !is.finite(value) || value != floor(value) || value < minimum ||
      value > .Machine$integer.max) {
    fail(sprintf("%s must be a native R integer", name))
  }
  as.integer(value)
}

scalar_number <- function(value, name, lower = -Inf, upper = Inf) {
  if (length(value) != 1L || is.logical(value) || !is.numeric(value) ||
      !is.finite(value) || value < lower || value > upper) {
    fail(sprintf("%s must be finite and in range", name))
  }
  as.numeric(value)
}

write_count_matrix <- function(value, path, gene_ids, cell_ids) {
  if (!is.matrix(value) || !identical(dim(value), c(length(gene_ids), length(cell_ids))) ||
      any(!is.finite(value)) || any(value < 0) || any(value != floor(value))) {
    fail(sprintf("invalid count matrix for %s", basename(path)))
  }
  connection <- file(path, open = "wb")
  on.exit(close(connection), add = TRUE)
  writeLines(paste(c("gene_id", cell_ids), collapse = "\t"), connection, useBytes = TRUE)
  for (index in seq_along(gene_ids)) {
    counts <- formatC(value[index, ], format = "f", digits = 0)
    writeLines(
      paste(c(gene_ids[index], counts), collapse = "\t"),
      connection,
      useBytes = TRUE
    )
  }
}

arguments <- commandArgs(trailingOnly = TRUE)
if (length(arguments) != 3L) {
  fail("usage: run_symsim.R CONFIG CHECKOUT OUTPUT_DIR")
}
config_path <- normalizePath(arguments[[1]], mustWork = TRUE)
checkout <- normalizePath(arguments[[2]], mustWork = TRUE)
output_dir <- normalizePath(arguments[[3]], mustWork = TRUE)
if (!dir.exists(checkout) || !dir.exists(output_dir)) {
  fail("checkout and output directory must exist")
}

if (!requireNamespace("digest", quietly = TRUE) ||
    !requireNamespace("jsonlite", quietly = TRUE) ||
    !requireNamespace("pkgload", quietly = TRUE)) {
  fail("digest, jsonlite and pkgload are required")
}
config <- jsonlite::fromJSON(config_path, simplifyVector = FALSE)
assert_names(config, c("adapter", "schema_version", "simulation", "seeds", "views"), "config")
if (!identical(config$schema_version, 1L)) {
  fail("unsupported config schema")
}
assert_names(
  config$adapter,
  c("python_adapter_sha256", "r_runner_sha256"),
  "adapter"
)
script_argument <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (length(script_argument) != 1L) {
  fail("R runner path is unavailable")
}
script_path <- normalizePath(sub("^--file=", "", script_argument), mustWork = TRUE)
runner_sha256 <- digest::digest(
  script_path,
  algo = "sha256",
  serialize = FALSE,
  file = TRUE
)
if (!identical(config$adapter$r_runner_sha256, runner_sha256) ||
    !grepl("^[0-9a-f]{64}$", config$adapter$python_adapter_sha256)) {
  fail("adapter byte commitment does not match the executing runner")
}

simulation_keys <- c(
  "cells", "genes", "gene_length", "gene_module_prop", "i_minpop",
  "marker_log2fc_threshold", "min_popsize", "n_de_evf", "nevf",
  "prop_hge", "vary"
)
assert_names(config$simulation, simulation_keys, "simulation")
cells <- scalar_integer(config$simulation$cells, "cells", 20L)
genes <- scalar_integer(config$simulation$genes, "genes", 20L)
gene_length <- scalar_integer(config$simulation$gene_length, "gene_length", 1L)
min_popsize <- scalar_integer(config$simulation$min_popsize, "min_popsize", 1L)
i_minpop <- scalar_integer(config$simulation$i_minpop, "i_minpop", 1L)
nevf <- scalar_integer(config$simulation$nevf, "nevf", 1L)
n_de_evf <- scalar_integer(config$simulation$n_de_evf, "n_de_evf", 0L)
marker_threshold <- scalar_number(
  config$simulation$marker_log2fc_threshold,
  "marker_log2fc_threshold"
)
if (cells %% 20L != 0L || min_popsize != cells %/% 20L || i_minpop != 1L ||
    nevf != 10L || n_de_evf != 9L ||
    !identical(config$simulation$vary, "s") ||
    scalar_number(config$simulation$prop_hge, "prop_hge") != 0 ||
    scalar_number(config$simulation$gene_module_prop, "gene_module_prop") != 0) {
  fail("simulation design differs from the prespecified SymSim panel")
}

assert_names(config$seeds, "biological", "seeds")
assert_names(config$seeds$biological, c("original", "mapped_r"), "biological seed")
biological_seed_r <- scalar_integer(
  config$seeds$biological$mapped_r,
  "biological mapped seed",
  1L
)
if (length(config$views) != 2L) {
  fail("exactly two technical views are required")
}
view_keys <- c(
  "technical_view", "measurement_seed_original", "measurement_seed_r",
  "protocol", "alpha_mean", "alpha_sd", "depth_mean", "depth_sd"
)
for (index in seq_along(config$views)) {
  assert_names(config$views[[index]], view_keys, sprintf("view %d", index))
}
view_names <- vapply(config$views, `[[`, character(1), "technical_view")
if (!identical(view_names, c("moderate", "severe"))) {
  fail("technical views must be moderate then severe")
}

pkgload::load_all(
  checkout,
  reset = TRUE,
  recompile = FALSE,
  export_all = FALSE,
  helpers = FALSE,
  attach = TRUE,
  quiet = TRUE
)
namespace <- asNamespace("SymSim")
simulate_true_counts <- get("SimulateTrueCounts", envir = namespace)
true_to_observed <- get("True2ObservedCounts", envir = namespace)
phyla5 <- get("Phyla5", envir = namespace)

set.seed(biological_seed_r)
phyla <- phyla5(plotting = FALSE)
simulate_true_counts_calls <- 0L
simulate_true_counts_calls <- simulate_true_counts_calls + 1L
true_result <- simulate_true_counts(
  ncells_total = cells,
  min_popsize = min_popsize,
  i_minpop = i_minpop,
  ngenes = genes,
  evf_type = "discrete",
  nevf = nevf,
  phyla = phyla,
  randseed = biological_seed_r,
  n_de_evf = n_de_evf,
  vary = "s",
  Sigma = 0.4,
  gene_module_prop = 0,
  prop_hge = 0
)
true_counts <- as.matrix(true_result$counts)
if (!identical(dim(true_counts), c(genes, cells)) ||
    any(!is.finite(true_counts)) || any(true_counts < 0) ||
    any(true_counts != floor(true_counts))) {
  fail("SimulateTrueCounts returned malformed true counts")
}
groups <- as.integer(true_result$cell_meta[, "pop"])
if (length(groups) != cells || !setequal(unique(groups), 1:5) ||
    sum(groups == 1L) != min_popsize) {
  fail("SimulateTrueCounts did not preserve the five-population design")
}

gene_lengths <- rep(gene_length, genes)
true2observed_counts_calls <- 0L
observed <- list()
measurement_seeds_r <- list()
for (view in config$views) {
  name <- view$technical_view
  measurement_seed_r <- scalar_integer(
    view$measurement_seed_r,
    sprintf("%s measurement seed", name),
    1L
  )
  if (!identical(view$protocol, "UMI")) {
    fail("SymSim validation views must use the UMI protocol")
  }
  set.seed(measurement_seed_r)
  true2observed_counts_calls <- true2observed_counts_calls + 1L
  result <- true_to_observed(
    true_counts = true_counts,
    meta_cell = true_result$cell_meta,
    protocol = "UMI",
    alpha_mean = scalar_number(view$alpha_mean, "alpha_mean", 0, 1),
    alpha_sd = scalar_number(view$alpha_sd, "alpha_sd", 0, 1),
    gene_len = gene_lengths,
    depth_mean = scalar_number(view$depth_mean, "depth_mean", 200),
    depth_sd = scalar_number(view$depth_sd, "depth_sd", 0)
  )
  counts <- as.matrix(result$counts)
  if (!identical(dim(counts), c(genes, cells)) || any(!is.finite(counts)) ||
      any(counts < 0) || any(counts != floor(counts)) || any(counts > true_counts)) {
    fail(sprintf("True2ObservedCounts returned malformed %s UMI counts", name))
  }
  observed[[name]] <- counts
  measurement_seeds_r[[name]] <- measurement_seed_r
}

kinetic <- true_result$kinetic_params
if (length(kinetic) != 3L || any(vapply(kinetic, function(x) {
  !is.matrix(x) || !identical(dim(x), c(genes, cells)) || any(!is.finite(x))
}, logical(1)))) {
  fail("SymSim kinetic parameters are malformed")
}
theoretical_mean <- kinetic[[3]] * kinetic[[1]] / (kinetic[[1]] + kinetic[[2]])
marker_scores <- matrix(0, nrow = genes, ncol = 5L)
marker_flags <- matrix(0L, nrow = genes, ncol = 5L)
for (group in 1:5) {
  group_mean <- rowMeans(theoretical_mean[, groups == group, drop = FALSE])
  other_mean <- rowMeans(theoretical_mean[, groups != group, drop = FALSE])
  score <- log2(pmax(group_mean, .Machine$double.xmin) /
                pmax(other_mean, .Machine$double.xmin))
  if (any(!is.finite(score))) {
    fail("non-finite theoretical marker score")
  }
  marker_scores[, group] <- score
  marker_flags[, group] <- as.integer(score > marker_threshold)
}

cell_width <- max(4L, nchar(as.character(cells)))
gene_width <- max(4L, nchar(as.character(genes)))
cell_ids <- sprintf(paste0("cell-%0", cell_width, "d"), seq_len(cells))
gene_ids <- sprintf(paste0("gene-%0", gene_width, "d"), seq_len(genes))
write_count_matrix(true_counts, file.path(output_dir, "true_counts.tsv"), gene_ids, cell_ids)
write_count_matrix(
  observed$moderate,
  file.path(output_dir, "observed_moderate.tsv"),
  gene_ids,
  cell_ids
)
write_count_matrix(
  observed$severe,
  file.path(output_dir, "observed_severe.tsv"),
  gene_ids,
  cell_ids
)

cell_connection <- file(file.path(output_dir, "cell_metadata.tsv"), open = "wb")
writeLines("cell_id\tgroup", cell_connection, useBytes = TRUE)
for (index in seq_along(cell_ids)) {
  writeLines(
    paste(cell_ids[index], groups[index], sep = "\t"),
    cell_connection,
    useBytes = TRUE
  )
}
close(cell_connection)

marker_columns <- "gene_id"
for (group in 1:5) {
  marker_columns <- c(
    marker_columns,
    sprintf("theoretical_log2fc_group_%d", group),
    sprintf("marker_group_%d", group)
  )
}
marker_connection <- file(file.path(output_dir, "marker_truth.tsv"), open = "wb")
writeLines(paste(marker_columns, collapse = "\t"), marker_connection, useBytes = TRUE)
for (gene in seq_along(gene_ids)) {
  fields <- gene_ids[gene]
  for (group in 1:5) {
    fields <- c(
      fields,
      sprintf("%.17g", marker_scores[gene, group]),
      as.character(marker_flags[gene, group])
    )
  }
  writeLines(paste(fields, collapse = "\t"), marker_connection, useBytes = TRUE)
}
close(marker_connection)

run_metadata <- list(
  schema_version = 1L,
  simulate_true_counts_calls = simulate_true_counts_calls,
  true2observed_counts_calls = true2observed_counts_calls,
  cells = cells,
  genes = genes,
  views = as.list(view_names),
  biological_seed_r = biological_seed_r,
  measurement_seeds_r = measurement_seeds_r,
  r_version = R.version.string,
  symsim_version = as.character(utils::packageVersion("SymSim"))
)
jsonlite::write_json(
  run_metadata,
  file.path(output_dir, "run_metadata.json"),
  auto_unbox = TRUE,
  pretty = FALSE,
  digits = NA,
  null = "null"
)
