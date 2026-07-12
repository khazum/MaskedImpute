#!/usr/bin/env Rscript

fail <- function(message) {
  stop(message, call. = FALSE)
}

assert_names <- function(value, expected, name) {
  observed <- names(value)
  if (is.null(observed) || length(observed) != length(expected) ||
      !setequal(observed, expected)) {
    fail(sprintf("%s has an invalid closed schema", name))
  }
}

scalar_integer <- function(value, name, minimum = 0L, maximum = .Machine$integer.max) {
  if (length(value) != 1L || is.logical(value) || !is.numeric(value) ||
      !is.finite(value) || value != floor(value) || value < minimum ||
      value > maximum) {
    fail(sprintf("%s must be a native R integer in range", name))
  }
  as.integer(value)
}

sha256_file <- function(path) {
  digest::digest(path, algo = "sha256", serialize = FALSE, file = TRUE)
}

canonicalize_json <- function(value) {
  if (is.list(value)) {
    if (!is.null(names(value))) {
      value <- value[sort(names(value), method = "radix")]
    }
    return(lapply(value, canonicalize_json))
  }
  value
}

write_canonical_json <- function(value, path) {
  payload <- jsonlite::toJSON(
    canonicalize_json(value),
    auto_unbox = TRUE,
    digits = NA,
    null = "null",
    pretty = FALSE
  )
  connection <- file(path, open = "wb")
  on.exit(close(connection), add = TRUE)
  writeLines(payload, connection, useBytes = TRUE)
}

write_matrix <- function(value, path, gene_ids, cell_ids, integer) {
  if (!is.matrix(value) ||
      !identical(dim(value), c(length(gene_ids), length(cell_ids))) ||
      any(!is.finite(value)) || any(value < 0) ||
      (integer && any(value != floor(value)))) {
    fail(sprintf("invalid matrix for %s", basename(path)))
  }
  connection <- file(path, open = "wb")
  on.exit(close(connection), add = TRUE)
  writeLines(
    paste(c("gene_id", cell_ids), collapse = "\t"),
    connection,
    useBytes = TRUE
  )
  for (index in seq_along(gene_ids)) {
    formatted <- if (integer) {
      formatC(value[index, ], format = "f", digits = 0)
    } else {
      sprintf("%.17g", value[index, ])
    }
    writeLines(
      paste(c(gene_ids[index], formatted), collapse = "\t"),
      connection,
      useBytes = TRUE
    )
  }
}

canonical_ids <- function(prefix, count) {
  width <- max(4L, nchar(as.character(count)))
  sprintf(paste0(prefix, "-%0", width, "d"), seq_len(count))
}

proportional_allocations <- function(cells, source_sizes) {
  total <- sum(source_sizes)
  allocations <- floor(cells * source_sizes / total)
  remainders <- (cells * source_sizes) %% total
  missing <- cells - sum(allocations)
  if (missing > 0L) {
    order <- order(-remainders, seq_along(source_sizes), method = "radix")
    allocations[order[seq_len(missing)]] <-
      allocations[order[seq_len(missing)]] + 1L
  }
  as.integer(allocations)
}

arguments <- commandArgs(trailingOnly = TRUE)
if (length(arguments) != 4L) {
  fail("usage: run_sparsim.R CONFIG CHECKOUT OUTPUT_DIR BUILD_CACHE")
}
config_path <- normalizePath(arguments[[1]], mustWork = TRUE)
checkout <- normalizePath(arguments[[2]], mustWork = TRUE)
output_dir <- normalizePath(arguments[[3]], mustWork = TRUE)
build_cache <- normalizePath(arguments[[4]], mustWork = TRUE)
if (!dir.exists(checkout) || !dir.exists(output_dir) || !dir.exists(build_cache)) {
  fail("checkout, output directory and build cache must exist")
}
if (!requireNamespace("digest", quietly = TRUE) ||
    !requireNamespace("jsonlite", quietly = TRUE) ||
    !requireNamespace("Rcpp", quietly = TRUE)) {
  fail("digest, jsonlite and Rcpp are required")
}

config_bytes <- readBin(config_path, what = "raw", n = file.info(config_path)$size)
config <- jsonlite::fromJSON(rawToChar(config_bytes), simplifyVector = FALSE)
assert_names(
  config,
  c(
    "adapter", "environment", "schema_version", "seeds", "simulation",
    "source", "views"
  ),
  "config"
)
if (!identical(config$schema_version, 1L)) {
  fail("unsupported SPARSim config schema")
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
if (!identical(config$adapter$r_runner_sha256, sha256_file(script_path)) ||
    !grepl("^[0-9a-f]{64}$", config$adapter$python_adapter_sha256)) {
  fail("adapter byte commitment does not match the executing runner")
}

assert_names(
  config$environment,
  c("compiler_executable_sha256", "environment_sha256"),
  "environment"
)
if (!grepl("^[0-9a-f]{64}$", config$environment$environment_sha256) ||
    !identical(
      config$environment$compiler_executable_sha256,
      sha256_file("/usr/bin/g++")
    )) {
  fail("SPARSim compiler or environment commitment is invalid")
}

assert_names(config$source, c("commit", "files", "tree"), "source")
if (!identical(
      config$source$commit,
      "4e7712fb236a92ce7c173da169c8a29cc2a9f0ef"
    ) ||
    !identical(
      config$source$tree,
      "5d66b28cc6afd8d68364f4205cc983c7f681e2fe"
    )) {
  fail("SPARSim source pin is invalid")
}
source_paths <- c(
  cpp = "src/Random_number.cpp",
  preset = "data/Chu_param_preset.RData",
  simulate = "R/SPARSim_simulate.R",
  utilities = "R/SPARSim_utilities.R"
)
assert_names(config$source$files, names(source_paths), "source files")
resolved_sources <- list()
for (role in names(source_paths)) {
  descriptor <- config$source$files[[role]]
  assert_names(descriptor, c("path", "sha256"), sprintf("source file %s", role))
  if (!identical(descriptor$path, unname(source_paths[[role]]))) {
    fail(sprintf("source path differs for %s", role))
  }
  path <- normalizePath(file.path(checkout, descriptor$path), mustWork = TRUE)
  checkout_prefix <- paste0(checkout, .Platform$file.sep)
  if (!startsWith(path, checkout_prefix) ||
      !identical(descriptor$sha256, sha256_file(path))) {
    fail(sprintf("source bytes differ for %s", role))
  }
  resolved_sources[[role]] <- path
}

simulation_keys <- c(
  "cells", "gene_selection", "gene_selection_domain", "genes",
  "group_allocations", "group_presets", "library_template_selection",
  "source_group_sizes"
)
assert_names(config$simulation, simulation_keys, "simulation")
cells <- scalar_integer(config$simulation$cells, "cells", 3L)
genes <- scalar_integer(config$simulation$genes, "genes", 1L, 17782L)
if (!identical(
      config$simulation$gene_selection,
      "sha256_ranked_source_gene_id_v1"
    ) ||
    !identical(
      config$simulation$gene_selection_domain,
      "maskimpute-sparsim-gene-v1"
    ) ||
    !identical(
      config$simulation$library_template_selection,
      "midpoint_quantile_with_replacement"
    )) {
  fail("SPARSim selection design changed")
}
expected_presets <- list(
  `chu-c1` = "Chu_C1",
  `chu-c3` = "Chu_C3",
  `chu-c6` = "Chu_C6"
)
expected_source_sizes <- list(`chu-c1` = 92L, `chu-c3` = 66L, `chu-c6` = 188L)
if (!identical(config$simulation$group_presets, expected_presets) ||
    !identical(config$simulation$source_group_sizes, expected_source_sizes)) {
  fail("SPARSim Chu group design changed")
}
assert_names(
  config$simulation$group_allocations,
  names(expected_presets),
  "group allocations"
)
source_sizes <- unlist(expected_source_sizes, use.names = TRUE)
expected_allocations <- proportional_allocations(cells, source_sizes)
names(expected_allocations) <- names(source_sizes)
observed_allocations <- vapply(
  config$simulation$group_allocations,
  scalar_integer,
  integer(1),
  name = "group allocation",
  minimum = 1L
)
if (!identical(observed_allocations, expected_allocations) ||
    sum(observed_allocations) != cells) {
  fail("SPARSim proportional Chu allocation changed")
}

assert_names(config$seeds, "biological", "seeds")
assert_names(config$seeds$biological, c("mapped_r", "original"), "biological seed")
biological_seed_r <- scalar_integer(
  config$seeds$biological$mapped_r,
  "biological mapped seed",
  1L
)
if (length(config$views) != 2L) {
  fail("exactly two SPARSim views are required")
}
view_keys <- c(
  "library_size_divisor", "library_size_rounding", "measurement_seed_original",
  "measurement_seed_r", "technical_view"
)
for (index in seq_along(config$views)) {
  assert_names(config$views[[index]], view_keys, sprintf("view %d", index))
}
view_names <- vapply(config$views, `[[`, character(1), "technical_view")
if (!identical(view_names, c("moderate", "severe"))) {
  fail("SPARSim views must be moderate then severe")
}
expected_divisors <- c(moderate = 100L, severe = 400L)
measurement_seeds_r <- list()
for (view in config$views) {
  name <- view$technical_view
  divisor <- scalar_integer(
    view$library_size_divisor,
    sprintf("%s library divisor", name),
    1L
  )
  if (divisor != expected_divisors[[name]] ||
      !identical(view$library_size_rounding, "nearest_half_up_minimum_1")) {
    fail(sprintf("SPARSim %s technical regime changed", name))
  }
  measurement_seeds_r[[name]] <- scalar_integer(
    view$measurement_seed_r,
    sprintf("%s measurement seed", name),
    1L
  )
}
if (length(unique(c(biological_seed_r, unlist(measurement_seeds_r)))) != 3L) {
  fail("SPARSim native seeds must be distinct")
}

Rcpp::sourceCpp(
  file = resolved_sources$cpp,
  rebuild = TRUE,
  cacheDir = file.path(build_cache, "rcpp-cache"),
  showOutput = FALSE,
  verbose = FALSE
)
source_cpp_calls <- 1L
sys.source(resolved_sources$utilities, envir = .GlobalEnv, keep.source = FALSE)
sys.source(resolved_sources$simulate, envir = .GlobalEnv, keep.source = FALSE)
preset_environment <- new.env(parent = emptyenv())
load(resolved_sources$preset, envir = preset_environment)
if (!exists("Chu_param_preset", envir = preset_environment, inherits = FALSE)) {
  fail("Chu preset object is unavailable")
}
Chu_param_preset <- get("Chu_param_preset", envir = preset_environment)
if (!identical(names(Chu_param_preset), paste0("Chu_C", seq_len(6L)))) {
  fail("Chu preset object has unexpected conditions")
}
selected_presets <- Chu_param_preset[unlist(expected_presets, use.names = FALSE)]
observed_source_sizes <- vapply(
  selected_presets,
  function(value) length(value$lib_size),
  integer(1)
)
if (!identical(unname(observed_source_sizes), unname(source_sizes))) {
  fail("Chu preset cell counts changed")
}
source_gene_ids <- names(selected_presets[[1]]$intensity)
if (length(source_gene_ids) != 17782L ||
    any(vapply(selected_presets, function(value) {
      !identical(names(value$intensity), source_gene_ids) ||
        length(value$variability) != length(source_gene_ids)
    }, logical(1)))) {
  fail("Chu preset gene identities changed")
}
gene_selection_hashes <- vapply(
  source_gene_ids,
  function(gene_id) digest::digest(
    paste(config$simulation$gene_selection_domain, gene_id, sep = "|"),
    algo = "sha256",
    serialize = FALSE
  ),
  character(1)
)
selection_order <- order(
  gene_selection_hashes,
  source_gene_ids,
  method = "radix"
)
selected_gene_ids <- source_gene_ids[selection_order[seq_len(genes)]]
if (any(!nzchar(selected_gene_ids)) || anyDuplicated(selected_gene_ids)) {
  fail("SPARSim selected source gene IDs are invalid")
}
cell_ids <- canonical_ids("cell", cells)
gene_ids <- canonical_ids("gene", genes)

build_parameters <- function(divisor) {
  result <- list()
  cell_offset <- 0L
  for (group in names(expected_presets)) {
    preset_name <- expected_presets[[group]]
    source <- selected_presets[[preset_name]]
    count <- observed_allocations[[group]]
    source_count <- length(source$lib_size)
    indices <- floor(((seq_len(count) - 0.5) * source_count) / count) + 1L
    chosen <- source$lib_size[indices]
    library_sizes <- pmax(1, floor(chosen / divisor + 0.5))
    ids <- cell_ids[cell_offset + seq_len(count)]
    names(library_sizes) <- ids
    parameter <- list(
      intensity = source$intensity[selected_gene_ids],
      variability = source$variability[selected_gene_ids],
      lib_size = library_sizes,
      name = group
    )
    result[[group]] <- parameter
    cell_offset <- cell_offset + count
  }
  result
}

sparsim_simulation_calls <- 0L
results <- list()
for (view in config$views) {
  name <- view$technical_view
  parameters <- build_parameters(view$library_size_divisor)
  sparsim_simulation_calls <- sparsim_simulation_calls + 1L
  result <- SPARSim_simulation(
    dataset_parameter = parameters,
    output_sim_param_matrices = FALSE,
    output_batch_matrix = FALSE,
    gene_expr_simulation_seed = biological_seed_r,
    count_data_simulation_seed = measurement_seeds_r[[name]],
    preserve_global_rng = TRUE
  )
  expected_libraries <- unlist(
    lapply(parameters, function(value) value$lib_size),
    use.names = TRUE
  )
  counts <- as.matrix(result$count_matrix)
  latent <- as.matrix(result$gene_matrix)
  if (!identical(dim(counts), c(genes, cells)) ||
      !identical(dim(latent), c(genes, cells)) ||
      any(!is.finite(counts)) || any(counts < 0) || any(counts != floor(counts)) ||
      any(!is.finite(latent)) || any(latent < 0) ||
      !identical(rownames(counts), selected_gene_ids) ||
      !identical(rownames(latent), selected_gene_ids) ||
      !identical(colnames(counts), cell_ids) ||
      !identical(colnames(latent), cell_ids) ||
      !identical(as.numeric(colSums(counts)), as.numeric(expected_libraries))) {
    fail(sprintf("SPARSim returned malformed %s matrices", name))
  }
  results[[name]] <- list(counts = counts, latent = latent)
}
gene_matrix_equal <- identical(results$moderate$latent, results$severe$latent)
if (!gene_matrix_equal) {
  fail("SPARSim paired views did not preserve the exact gene_matrix")
}
if (identical(results$moderate$counts, results$severe$counts)) {
  fail("SPARSim paired views produced identical measured counts")
}

write_matrix(
  results$moderate$latent,
  file.path(output_dir, "latent_expression.tsv"),
  gene_ids,
  cell_ids,
  integer = FALSE
)
write_matrix(
  results$moderate$counts,
  file.path(output_dir, "observed_moderate.tsv"),
  gene_ids,
  cell_ids,
  integer = TRUE
)
write_matrix(
  results$severe$counts,
  file.path(output_dir, "observed_severe.tsv"),
  gene_ids,
  cell_ids,
  integer = TRUE
)

groups <- rep(names(observed_allocations), observed_allocations)
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

gene_connection <- file(file.path(output_dir, "gene_metadata.tsv"), open = "wb")
writeLines("gene_id\tsource_gene_id", gene_connection, useBytes = TRUE)
for (index in seq_along(gene_ids)) {
  writeLines(
    paste(gene_ids[index], selected_gene_ids[index], sep = "\t"),
    gene_connection,
    useBytes = TRUE
  )
}
close(gene_connection)

matrix_names <- c(
  "latent_expression.tsv", "observed_moderate.tsv", "observed_severe.tsv"
)
array_sha256 <- as.list(vapply(
  matrix_names,
  function(name) sha256_file(file.path(output_dir, name)),
  character(1)
))
run_metadata <- list(
  array_sha256 = array_sha256,
  biological_seed_r = biological_seed_r,
  cells = cells,
  compiler_sha256 = config$environment$compiler_executable_sha256,
  config_sha256 = sha256_file(config_path),
  gene_matrix_equal = gene_matrix_equal,
  genes = genes,
  group_allocations = as.list(observed_allocations),
  measurement_seeds_r = measurement_seeds_r,
  r_version = R.version.string,
  rcpp_version = as.character(utils::packageVersion("Rcpp")),
  schema_version = 1L,
  source_cpp_calls = source_cpp_calls,
  sparsim_simulation_calls = sparsim_simulation_calls,
  views = as.list(view_names)
)
write_canonical_json(
  run_metadata,
  file.path(output_dir, "run_metadata.json")
)
