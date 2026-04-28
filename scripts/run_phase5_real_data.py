#!/usr/bin/env python3
"""Robust real-data imputation and label-free clustering evaluation.

The script implements the Phase 5 fairness upgrades:
  * label-free HVG selection from original/full matrices rather than marker-gene subsets;
  * a PCA/kNN/Leiden robustness grid;
  * repeated random seeds and matched-cluster-count summaries;
  * batch-association diagnostics when batch metadata are available;
  * PBMC68k support from the 10x RDS artifact and annotations; and
  * DCA wall-clock timeouts recorded as method status instead of silent omission.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import io as scipy_io
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from rds2py import read_rds

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_imputation as ri


DISPLAY_NAMES = {
    "baseline": "Observed",
    "dca": "DCA",
    "scvi": "scVI",
    "alra": "ALRA",
    "magic": "MAGIC",
    "maskimpute": "MaskImpute",
}
STOCHASTIC_IMPUTERS = {"alra", "scvi", "maskimpute"}
MISSING_VALUES = {"", "NA", "NaN", "nan", "None", "none", "null"}


@dataclass
class RealDataset:
    name: str
    path: str
    counts: np.ndarray
    logcounts: np.ndarray
    labels: np.ndarray
    label_name: str
    batch: Optional[np.ndarray]
    batch_name: str
    target_sum: float
    feature_mode: str
    n_features_total: int
    n_features_used: int
    n_cells_total: int
    n_cells_used: int
    source_note: str


def _as_str_array(values: object) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return np.array([str(x) for x in arr], dtype=object)


def _valid_label_mask(labels: np.ndarray) -> np.ndarray:
    return np.array([str(x) not in MISSING_VALUES for x in labels], dtype=bool)


def _get_label_column(sce, preferred: Sequence[str]) -> Tuple[str, np.ndarray]:
    for name in preferred:
        vals = ri._get_coldata_column(sce, name)
        if vals is None:
            continue
        labels = _as_str_array(vals)
        ok = _valid_label_mask(labels)
        if ok.sum() >= 2 and len(set(labels[ok])) >= 2:
            return name, labels
    raise RuntimeError(f"No usable label column found. Tried: {', '.join(preferred)}")


def _get_batch_column(sce, preferred: Sequence[str], labels: np.ndarray) -> Tuple[str, Optional[np.ndarray]]:
    label_values = np.asarray(labels, dtype=object)
    for name in preferred:
        vals = ri._get_coldata_column(sce, name)
        if vals is None:
            continue
        batch = _as_str_array(vals)
        ok = _valid_label_mask(batch)
        uniq = set(batch[ok])
        if len(uniq) < 2:
            continue
        if batch.shape == label_values.shape and np.array_equal(batch, label_values):
            continue
        return name, batch
    return "", None


def _cell_names_from_sce(sce) -> List[str]:
    for attr in ("colnames", "col_names", "column_names"):
        try:
            values = getattr(sce, attr)
            if callable(values):
                values = values()
            names = [str(x) for x in values]
            if names:
                return names
        except Exception:
            continue
    return []


def _derive_batch_from_cell_names(cell_names: Sequence[str]) -> Tuple[str, Optional[np.ndarray]]:
    if not cell_names:
        return "", None
    names = np.asarray([str(x) for x in cell_names], dtype=object)
    if all("_" in x for x in names):
        batch = np.array([x.split("_", 1)[0] for x in names], dtype=object)
        if 1 < len(set(batch)) < len(batch) * 0.8:
            return "cell_id_prefix", batch
    if all("-" in x for x in names):
        batch = np.array(["gem_group_" + x.rsplit("-", 1)[1] for x in names], dtype=object)
        if 1 < len(set(batch)) < len(batch) * 0.8:
            return "barcode_suffix", batch
    return "", None


def _to_csr_cells_by_genes(matrix: object) -> sparse.csr_matrix:
    if sparse.issparse(matrix):
        return matrix.T.tocsr()
    arr = np.asarray(matrix)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D assay, got shape {arr.shape}")
    return sparse.csr_matrix(arr.T)


def select_hvgs_from_sparse(
    counts: sparse.spmatrix,
    logcounts: sparse.spmatrix,
    *,
    n_hvg: int,
    min_cells: int,
) -> np.ndarray:
    counts = counts.tocsr()
    logcounts = logcounts.tocsr()
    n_cells, n_genes = counts.shape
    if n_hvg <= 0 or n_hvg >= n_genes:
        return np.arange(n_genes, dtype=int)
    expr_cells = np.asarray((counts > 0).sum(axis=0)).reshape(-1)
    mean = np.asarray(logcounts.mean(axis=0)).reshape(-1)
    mean_sq = np.asarray(logcounts.multiply(logcounts).mean(axis=0)).reshape(-1)
    var = np.maximum(mean_sq - mean * mean, 0.0)
    disp = var / np.maximum(mean, 1e-8)
    valid = np.where((expr_cells >= min_cells) & np.isfinite(disp) & (mean > 0))[0]
    if valid.size == 0:
        raise ValueError("No valid genes available for HVG selection")
    if valid.size <= n_hvg:
        return valid.astype(int)

    order = np.argsort(mean[valid], kind="mergesort")
    bins = np.array_split(order, min(20, valid.size))
    score = np.full(valid.size, -np.inf, dtype=np.float64)
    log_disp = np.log1p(np.maximum(disp[valid], 0.0))
    for idx in bins:
        vals = log_disp[idx]
        center = np.nanmedian(vals)
        mad = np.nanmedian(np.abs(vals - center))
        if not np.isfinite(mad) or mad <= 0:
            mad = np.nanstd(vals)
        if not np.isfinite(mad) or mad <= 0:
            mad = 1.0
        score[idx] = (vals - center) / mad
    chosen = valid[np.argsort(score)[::-1][:n_hvg]]
    return np.sort(chosen.astype(int))


def _subset_cells(
    counts: sparse.spmatrix,
    logcounts: sparse.spmatrix,
    labels: np.ndarray,
    batch: Optional[np.ndarray],
    *,
    max_cells: int,
    seed: int,
    strategy: str,
) -> Tuple[sparse.spmatrix, sparse.spmatrix, np.ndarray, Optional[np.ndarray], int]:
    n = labels.shape[0]
    if max_cells <= 0 or n <= max_cells or strategy == "none":
        return counts, logcounts, labels, batch, n
    rng = np.random.default_rng(seed)
    if strategy == "stratified":
        selected: List[int] = []
        for label in sorted(set(labels.tolist())):
            idx = np.where(labels == label)[0]
            quota = max(1, int(round(max_cells * idx.size / n)))
            selected.extend(rng.choice(idx, size=min(idx.size, quota), replace=False).tolist())
        if len(selected) < max_cells:
            rest = np.setdiff1d(np.arange(n), np.asarray(selected, dtype=int), assume_unique=False)
            selected.extend(rng.choice(rest, size=min(rest.size, max_cells - len(selected)), replace=False).tolist())
        idx = np.sort(np.asarray(selected[:max_cells], dtype=int))
    else:
        idx = np.sort(rng.choice(n, size=max_cells, replace=False))
    batch_sub = None if batch is None else batch[idx]
    return counts[idx], logcounts[idx], labels[idx], batch_sub, n


def load_sce_dataset(
    name: str,
    path: Path,
    *,
    label_columns: Sequence[str],
    batch_columns: Sequence[str],
    n_hvg: int,
    min_hvg_cells: int,
    max_cells: int,
    cell_sample_strategy: str,
    seed: int,
) -> RealDataset:
    sce = read_rds(str(path))
    if not hasattr(sce, "assay"):
        raise TypeError(f"Unsupported RDS object: {type(sce)}")
    counts_sparse = _to_csr_cells_by_genes(sce.assay("counts"))
    log_sparse = _to_csr_cells_by_genes(sce.assay("logcounts"))
    if counts_sparse.shape != log_sparse.shape:
        raise ValueError(f"counts/logcounts shape mismatch: {counts_sparse.shape} vs {log_sparse.shape}")

    label_name, labels = _get_label_column(sce, label_columns)
    batch_name, batch = _get_batch_column(sce, batch_columns, labels)
    if batch is None:
        batch_name, batch = _derive_batch_from_cell_names(_cell_names_from_sce(sce))

    ok = _valid_label_mask(labels)
    counts_sparse = counts_sparse[ok]
    log_sparse = log_sparse[ok]
    labels = labels[ok]
    if batch is not None:
        batch = batch[ok]

    counts_sparse, log_sparse, labels, batch, n_before_sample = _subset_cells(
        counts_sparse,
        log_sparse,
        labels,
        batch,
        max_cells=max_cells,
        seed=seed,
        strategy=cell_sample_strategy,
    )
    hvg_idx = select_hvgs_from_sparse(counts_sparse, log_sparse, n_hvg=n_hvg, min_cells=min_hvg_cells)
    counts = counts_sparse[:, hvg_idx].toarray().astype(np.float32, copy=False)
    logcounts = log_sparse[:, hvg_idx].toarray().astype(np.float32, copy=False)
    norm_info = ri.get_normalization_info(sce)
    return RealDataset(
        name=name,
        path=str(path),
        counts=counts,
        logcounts=logcounts,
        labels=labels,
        label_name=label_name,
        batch=batch,
        batch_name=batch_name,
        target_sum=float(norm_info["target_sum"]),
        feature_mode="label_free_hvg",
        n_features_total=int(counts_sparse.shape[1]),
        n_features_used=int(hvg_idx.size),
        n_cells_total=int(n_before_sample),
        n_cells_used=int(labels.shape[0]),
        source_note="SingleCellExperiment original matrix",
    )


def _run_pbmc_extractor(
    *,
    input_path: Path,
    annotations: Path,
    cache_dir: Path,
    n_hvg: int,
    max_cells: int,
    seed: int,
    sampling: str,
    target_sum: float,
    min_cells: int,
    rscript_bin: str,
) -> None:
    counts_path = cache_dir / "counts.mtx"
    log_path = cache_dir / "logcounts.mtx"
    meta_path = cache_dir / "metadata.tsv"
    if counts_path.exists() and log_path.exists() and meta_path.exists():
        return
    cmd = [
        str(Path(rscript_bin).expanduser()),
        str(Path(__file__).resolve().parent / "extract_pbmc68k_hvg.R"),
        "--input",
        str(input_path),
        "--annotations",
        str(annotations),
        "--out-dir",
        str(cache_dir),
        "--n-hvg",
        str(n_hvg),
        "--max-cells",
        str(max_cells),
        "--seed",
        str(seed),
        "--sampling",
        sampling,
        "--target-sum",
        str(target_sum),
        "--min-cells",
        str(min_cells),
    ]
    subprocess.run(cmd, check=True)


def load_pbmc68k_dataset(
    name: str,
    path: Path,
    *,
    annotations: Path,
    cache_root: Path,
    n_hvg: int,
    max_cells: int,
    seed: int,
    cell_sample_strategy: str,
    target_sum: float,
    min_hvg_cells: int,
    rscript_bin: str,
) -> RealDataset:
    if max_cells <= 0:
        cell_tag = "all"
    else:
        cell_tag = str(max_cells)
    cache_dir = cache_root / f"pbmc68k_hvg{n_hvg}_cells{cell_tag}_seed{seed}_{cell_sample_strategy}"
    _run_pbmc_extractor(
        input_path=path,
        annotations=annotations,
        cache_dir=cache_dir,
        n_hvg=n_hvg,
        max_cells=max_cells,
        seed=seed,
        sampling=cell_sample_strategy,
        target_sum=target_sum,
        min_cells=min_hvg_cells,
        rscript_bin=rscript_bin,
    )
    counts = scipy_io.mmread(cache_dir / "counts.mtx").tocsr().toarray().astype(np.float32, copy=False)
    logcounts = scipy_io.mmread(cache_dir / "logcounts.mtx").tocsr().toarray().astype(np.float32, copy=False)
    meta = pd.read_csv(cache_dir / "metadata.tsv", sep="\t")
    norm = pd.read_csv(cache_dir / "normalization.tsv", sep="\t", header=None, names=["key", "value"])
    norm_map = dict(zip(norm["key"].astype(str), norm["value"].astype(str)))
    return RealDataset(
        name=name,
        path=str(path),
        counts=counts,
        logcounts=logcounts,
        labels=meta["label"].astype(str).to_numpy(dtype=object),
        label_name="celltype",
        batch=meta["batch"].astype(str).to_numpy(dtype=object) if "batch" in meta else None,
        batch_name="barcode_suffix",
        target_sum=float(norm_map.get("target_sum", target_sum)),
        feature_mode="label_free_hvg",
        n_features_total=32738,
        n_features_used=int(counts.shape[1]),
        n_cells_total=68579,
        n_cells_used=int(counts.shape[0]),
        source_note="10x PBMC68k matrix with published barcode annotations",
    )


def load_dataset(spec: Tuple[str, Path], args: argparse.Namespace) -> RealDataset:
    name, path = spec
    lower = path.name.lower()
    if "pbmc68k" in lower or lower == "pbmc68k_data.rds":
        return load_pbmc68k_dataset(
            name,
            path,
            annotations=Path(args.pbmc_annotations),
            cache_root=Path(args.cache_dir),
            n_hvg=args.n_hvg,
            max_cells=args.pbmc_max_cells,
            seed=args.seed,
            cell_sample_strategy=args.cell_sample_strategy,
            target_sum=args.target_sum,
            min_hvg_cells=args.min_hvg_cells,
            rscript_bin=args.rscript_bin,
        )
    return load_sce_dataset(
        name,
        path,
        label_columns=[x.strip() for x in args.label_columns.split(",") if x.strip()],
        batch_columns=[x.strip() for x in args.batch_columns.split(",") if x.strip()],
        n_hvg=args.n_hvg,
        min_hvg_cells=args.min_hvg_cells,
        max_cells=args.max_cells,
        cell_sample_strategy=args.cell_sample_strategy,
        seed=args.seed,
    )


def run_method(
    method: str,
    data: RealDataset,
    *,
    seed: int,
    n_jobs: int,
    dca_bin: str,
    dca_threads: int,
    dca_timeout_sec: Optional[float],
    scvi_epochs: int,
    scvi_batch_size: int,
    alra_max_rank: int,
) -> np.ndarray:
    if method == "baseline":
        return np.asarray(data.logcounts, dtype=np.float32)
    if method == "magic":
        return ri.run_magic(data.logcounts, n_jobs=n_jobs)
    if method == "alra":
        return ri.run_alra(data.logcounts, seed=seed, rank=None, max_rank=alra_max_rank, n_iter=7)
    if method == "maskimpute":
        return ri.run_maskimpute(data.logcounts, data.counts, seed=seed)
    if method == "scvi":
        return ri.run_scvi(
            data.counts,
            data.target_sum,
            seed=seed,
            max_epochs=scvi_epochs,
            n_latent=10,
            n_hidden=128,
            n_layers=2,
            batch_size=scvi_batch_size,
        )
    if method == "dca":
        old_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        try:
            counts_imp = ri.run_dca(
                data.counts,
                dca_bin=dca_bin,
                ae_type=None,
                epochs=None,
                batch_size=None,
                threads=dca_threads,
                ridge=None,
                verbose=False,
                timeout_sec=dca_timeout_sec,
            )
        finally:
            if old_cuda is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = old_cuda
        return ri.normalize_counts_to_logcounts(counts_imp, data.target_sum)
    raise ValueError(f"Unknown method: {method}")


def _snn_graph(embedding: np.ndarray, n_neighbors: int):
    import igraph as ig

    n_cells = embedding.shape[0]
    k = int(max(2, min(n_neighbors, n_cells - 1)))
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", n_jobs=1)
    nn.fit(embedding)
    indices = nn.kneighbors(embedding, return_distance=False)[:, 1:]
    neighbor_sets = [set(map(int, row)) for row in indices]
    edge_weights: Dict[Tuple[int, int], float] = {}
    for i, row in enumerate(indices):
        si = neighbor_sets[i]
        for j_raw in row:
            j = int(j_raw)
            if i == j:
                continue
            a, b = (i, j) if i < j else (j, i)
            sj = neighbor_sets[j]
            shared = len(si.intersection(sj))
            union = len(si.union(sj))
            weight = shared / union if union else 0.0
            if weight <= 0.0:
                weight = 1.0 / (2.0 * k)
            old = edge_weights.get((a, b))
            if old is None or weight > old:
                edge_weights[(a, b)] = weight
    graph = ig.Graph(n=n_cells, edges=list(edge_weights.keys()), directed=False)
    graph.es["weight"] = list(edge_weights.values())
    return graph


def _leiden_membership(graph, resolution: float, seed: int) -> np.ndarray:
    import leidenalg

    part = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=float(resolution),
        seed=int(seed),
    )
    return np.asarray(part.membership, dtype=int)


def _safe_silhouette(embedding: np.ndarray, groups: Optional[np.ndarray], seed: int, sample_size: int) -> float:
    if groups is None:
        return float("nan")
    labels = np.asarray(groups, dtype=object)
    ok = _valid_label_mask(labels)
    labels = labels[ok]
    if labels.size < 3 or len(set(labels.tolist())) < 2 or len(set(labels.tolist())) >= labels.size:
        return float("nan")
    emb = embedding[ok]
    size = min(sample_size, emb.shape[0]) if sample_size and sample_size > 0 else None
    try:
        return float(silhouette_score(emb, labels, metric="euclidean", sample_size=size, random_state=seed))
    except Exception:
        return float("nan")


def cluster_grid(
    matrix: np.ndarray,
    labels: np.ndarray,
    batch: Optional[np.ndarray],
    *,
    n_pcs_values: Sequence[int],
    n_neighbors_values: Sequence[int],
    resolutions: Sequence[float],
    seed: int,
    silhouette_sample_size: int,
) -> List[Dict[str, object]]:
    x = np.asarray(matrix, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    n_cells, n_genes = x.shape
    if n_cells < 3 or n_genes < 2:
        raise ValueError(f"Need at least 3 cells and 2 genes, got {x.shape}")
    max_pcs_req = max(int(v) for v in n_pcs_values)
    max_pcs = int(max(2, min(max_pcs_req, n_cells - 1, n_genes - 1)))
    x_scaled = StandardScaler(with_mean=True, with_std=True).fit_transform(x)
    emb_full = PCA(n_components=max_pcs, svd_solver="randomized", random_state=seed).fit_transform(x_scaled)

    rows: List[Dict[str, object]] = []
    label_ok = _valid_label_mask(labels)
    y = labels[label_ok]
    for n_pcs_raw in n_pcs_values:
        n_pcs = int(max(2, min(int(n_pcs_raw), emb_full.shape[1])))
        emb = emb_full[:, :n_pcs]
        batch_sil = _safe_silhouette(emb, batch, seed, silhouette_sample_size)
        for k_raw in n_neighbors_values:
            k = int(max(2, min(int(k_raw), n_cells - 1)))
            graph = _snn_graph(emb, k)
            for res in resolutions:
                t0 = time.perf_counter()
                clusters = _leiden_membership(graph, float(res), seed)
                cluster_runtime = time.perf_counter() - t0
                c = clusters[label_ok]
                ari = float(adjusted_rand_score(y, c)) if y.size >= 2 else float("nan")
                nmi = float(normalized_mutual_info_score(y, c)) if y.size >= 2 else float("nan")
                batch_ari = batch_nmi = float("nan")
                if batch is not None:
                    b = np.asarray(batch, dtype=object)
                    batch_ok = _valid_label_mask(b)
                    if batch_ok.sum() >= 2 and len(set(b[batch_ok].tolist())) >= 2:
                        batch_ari = float(adjusted_rand_score(b[batch_ok], clusters[batch_ok]))
                        batch_nmi = float(normalized_mutual_info_score(b[batch_ok], clusters[batch_ok]))
                rows.append(
                    {
                        "seed": seed,
                        "n_pcs": n_pcs,
                        "n_neighbors": k,
                        "resolution": float(res),
                        "n_clusters": int(len(set(clusters.tolist()))),
                        "ari": ari,
                        "nmi": nmi,
                        "batch_ari": batch_ari,
                        "batch_nmi": batch_nmi,
                        "batch_silhouette": batch_sil,
                        "cluster_runtime_sec": cluster_runtime,
                    }
                )
    return rows


def parse_dataset_arg(raw: str) -> Tuple[str, Path]:
    if "=" in raw:
        name, path = raw.split("=", 1)
        return name.strip(), Path(path.strip())
    p = Path(raw)
    stem = p.stem.replace("_top1000markers", "")
    return stem, p


def parse_int_list(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_float_list(raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _fmt_mean_sd(mean: object, sd: object, digits: int = 3) -> str:
    try:
        m = float(mean)
    except Exception:
        return "--"
    if not np.isfinite(m):
        return "--"
    try:
        s = float(sd)
    except Exception:
        s = float("nan")
    if np.isfinite(s) and s > 0:
        return f"{m:.{digits}f} $\\pm$ {s:.{digits}f}"
    return f"{m:.{digits}f}"


def summarize(grid_df: pd.DataFrame, runs_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    all_keys = runs_df[["dataset", "method"]].drop_duplicates()
    for _, key in all_keys.iterrows():
        ds = key["dataset"]
        method = key["method"]
        run_sub = runs_df[(runs_df["dataset"] == ds) & (runs_df["method"] == method)]
        grid_sub = grid_df[(grid_df["dataset"] == ds) & (grid_df["method"] == method)] if not grid_df.empty else pd.DataFrame()
        base: Dict[str, object] = {
            "dataset": ds,
            "method": method,
            "display_method": DISPLAY_NAMES.get(method, method),
            "status": ";".join(sorted(set(str(x) for x in run_sub["status"] if str(x)))) or "ok",
            "n_cells": int(run_sub["n_cells"].iloc[0]),
            "n_genes": int(run_sub["n_genes"].iloc[0]),
            "n_labels": int(run_sub["n_labels"].iloc[0]),
            "n_seeds_requested": int(run_sub["n_seeds_requested"].iloc[0]),
            "n_method_runs": int(len(run_sub)),
            "impute_runtime_sec_mean": pd.to_numeric(run_sub["impute_runtime_sec"], errors="coerce").mean(),
            "error": " | ".join(sorted(set(str(x) for x in run_sub["error"] if str(x) and str(x) != "nan"))),
        }
        if grid_sub.empty:
            rows.append(base)
            continue
        for metric in ["ari", "nmi", "batch_ari", "batch_nmi", "batch_silhouette"]:
            vals = pd.to_numeric(grid_sub[metric], errors="coerce")
            base[f"{metric}_grid_mean"] = vals.mean()
            base[f"{metric}_grid_sd"] = vals.std(ddof=1)
            base[f"{metric}_oracle_best"] = vals.max() if metric in {"ari", "nmi"} else vals.min()
        base["n_clusters_grid_mean"] = pd.to_numeric(grid_sub["n_clusters"], errors="coerce").mean()
        base["n_grid_rows"] = int(len(grid_sub))

        matched_rows = []
        for seed, seed_sub in grid_sub.groupby("seed"):
            n_labels = int(seed_sub["n_labels"].iloc[0])
            tmp = seed_sub.copy()
            tmp["cluster_delta"] = (pd.to_numeric(tmp["n_clusters"], errors="coerce") - n_labels).abs()
            tmp["res_delta"] = (pd.to_numeric(tmp["resolution"], errors="coerce") - 1.0).abs()
            tmp = tmp.sort_values(["cluster_delta", "res_delta", "n_pcs", "n_neighbors"])
            matched_rows.append(tmp.iloc[0])
        matched = pd.DataFrame(matched_rows)
        for metric in ["ari", "nmi", "batch_ari", "batch_nmi", "batch_silhouette"]:
            vals = pd.to_numeric(matched[metric], errors="coerce")
            base[f"{metric}_matched_mean"] = vals.mean()
            base[f"{metric}_matched_sd"] = vals.std(ddof=1)
        base["n_clusters_matched_mean"] = pd.to_numeric(matched["n_clusters"], errors="coerce").mean()
        rows.append(base)
    return pd.DataFrame(rows)


def write_tex_summary(summary: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    methods = ["baseline", "dca", "scvi", "alra", "magic", "maskimpute"]
    lines = [
        "\\begin{tabular}{@{}llccccc@{}}",
        "\\toprule",
        "Dataset & Method & ARI grid & NMI grid & ARI matched & NMI matched & Batch ARI \\\\",
        "\\midrule",
    ]
    for ds in summary["dataset"].drop_duplicates().tolist():
        ds_rows = summary[summary["dataset"] == ds]
        for method in methods:
            row_df = ds_rows[ds_rows["method"] == method]
            if row_df.empty:
                continue
            row = row_df.iloc[0]
            status = str(row.get("status", ""))
            escaped_ds = str(ds).replace("_", "\\_")
            display = DISPLAY_NAMES.get(method, method)
            if "timeout" in status.lower() or pd.isna(row.get("ari_grid_mean")):
                note = "timeout" if "timeout" in status.lower() else "failed"
                lines.append(f"{escaped_ds} & {display} & \\multicolumn{{5}}{{c}}{{{note}}} \\\\")
                continue
            lines.append(
                f"{escaped_ds} & {display} & "
                f"{_fmt_mean_sd(row.get('ari_grid_mean'), row.get('ari_grid_sd'))} & "
                f"{_fmt_mean_sd(row.get('nmi_grid_mean'), row.get('nmi_grid_sd'))} & "
                f"{_fmt_mean_sd(row.get('ari_matched_mean'), row.get('ari_matched_sd'))} & "
                f"{_fmt_mean_sd(row.get('nmi_matched_mean'), row.get('nmi_matched_sd'))} & "
                f"{_fmt_mean_sd(row.get('batch_ari_grid_mean'), row.get('batch_ari_grid_sd'))} \\\\"
            )
        lines.append("\\addlinespace")
    if lines[-1] == "\\addlinespace":
        lines.pop()
    lines += ["\\bottomrule", "\\end{tabular}"]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def default_dataset_specs() -> List[str]:
    specs = [
        "Baron=datasets_original/baron-human.rds",
        "Zeisel=datasets_original/zeisel.rds",
    ]
    if Path("temp/pbmc68k_data.rds").exists():
        specs.append("PBMC68k=temp/pbmc68k_data.rds")
    return specs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", default=[], help="Dataset spec name=path or path. Can be repeated.")
    parser.add_argument("--methods", default="baseline,dca,scvi,alra,magic,maskimpute")
    parser.add_argument("--out-dir", default="results_real_data")
    parser.add_argument("--cache-dir", default="results_real_data/cache")
    parser.add_argument("--paper-table", default="paper/generated/real_data_table.tex")
    parser.add_argument("--label-columns", default="cell_type1,cell_type,CellType,label,labels,clust_id")
    parser.add_argument("--batch-columns", default="human,donor,batch,sample,Sample,Batch,orig.ident")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--n-hvg", type=int, default=1000)
    parser.add_argument("--min-hvg-cells", type=int, default=10)
    parser.add_argument("--max-cells", type=int, default=0, help="Optional max cells for non-PBMC datasets; 0 keeps all cells.")
    parser.add_argument("--pbmc-max-cells", type=int, default=12000, help="PBMC68k cell subsample for tractable repeated evaluation; 0 keeps all cells.")
    parser.add_argument("--cell-sample-strategy", choices=["none", "random", "stratified"], default="random")
    parser.add_argument("--target-sum", type=float, default=10000.0)
    parser.add_argument("--n-pcs-grid", default="20,50,100")
    parser.add_argument("--n-neighbors-grid", default="10,15,30")
    parser.add_argument("--resolution-grid", default="0.2,0.5,1.0,1.5,2.0")
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--dca-bin", default="~/miniconda3/envs/dca_env/bin/dca")
    parser.add_argument("--dca-threads", type=int, default=8)
    parser.add_argument("--dca-timeout-sec", type=float, default=600.0)
    parser.add_argument("--scvi-epochs", type=int, default=400)
    parser.add_argument("--scvi-batch-size", type=int, default=256)
    parser.add_argument("--alra-max-rank", type=int, default=100)
    parser.add_argument("--silhouette-sample-size", type=int, default=5000)
    parser.add_argument("--pbmc-annotations", default="temp/single-cell-3prime-paper/pbmc68k_analysis/68k_pbmc_barcodes_annotation.tsv")
    parser.add_argument("--rscript-bin", default="~/miniconda3/envs/r45_bio/bin/Rscript")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_specs = [parse_dataset_arg(spec) for spec in (args.dataset or default_dataset_specs())]
    methods = [m.strip().lower() for m in args.methods.split(",") if m.strip()]
    seeds = parse_int_list(args.seeds)
    n_pcs_values = parse_int_list(args.n_pcs_grid)
    n_neighbors_values = parse_int_list(args.n_neighbors_grid)
    resolutions = parse_float_list(args.resolution_grid)

    grid_rows: List[Dict[str, object]] = []
    run_rows: List[Dict[str, object]] = []

    for dataset_spec in dataset_specs:
        data = load_dataset(dataset_spec, args)
        n_labels = len(set(data.labels.tolist()))
        n_batches = 0 if data.batch is None else len(set(data.batch.tolist()))
        print(
            f"\n=== {data.name} ===\n"
            f"source={data.path}\n"
            f"cells={data.n_cells_used}/{data.n_cells_total} genes={data.n_features_used}/{data.n_features_total} "
            f"label={data.label_name} n_labels={n_labels} batch={data.batch_name or 'none'} n_batches={n_batches}",
            flush=True,
        )

        for method in methods:
            impute_seeds = seeds if method in STOCHASTIC_IMPUTERS else [seeds[0]]
            method_failed = False
            cached_imp: Optional[np.ndarray] = None
            for impute_seed in impute_seeds:
                print(f"  -> {method} impute_seed={impute_seed}", flush=True)
                status = "ok"
                error = ""
                log_imp: Optional[np.ndarray] = None
                t0 = time.perf_counter()
                try:
                    if method in STOCHASTIC_IMPUTERS or cached_imp is None:
                        log_imp = run_method(
                            method,
                            data,
                            seed=impute_seed,
                            n_jobs=args.n_jobs,
                            dca_bin=args.dca_bin,
                            dca_threads=args.dca_threads,
                            dca_timeout_sec=args.dca_timeout_sec,
                            scvi_epochs=args.scvi_epochs,
                            scvi_batch_size=args.scvi_batch_size,
                            alra_max_rank=args.alra_max_rank,
                        )
                        if method not in STOCHASTIC_IMPUTERS:
                            cached_imp = log_imp
                    else:
                        log_imp = cached_imp
                    if log_imp.shape != data.logcounts.shape:
                        raise ValueError(f"output shape {log_imp.shape} does not match input {data.logcounts.shape}")
                except TimeoutError as exc:
                    status = "timeout"
                    error = str(exc)
                    method_failed = True
                except Exception as exc:
                    status = "error"
                    error = str(exc)
                    method_failed = True
                impute_runtime = time.perf_counter() - t0
                run_rows.append(
                    {
                        "dataset": data.name,
                        "path": data.path,
                        "method": method,
                        "display_method": DISPLAY_NAMES.get(method, method),
                        "impute_seed": impute_seed,
                        "status": status,
                        "error": error,
                        "impute_runtime_sec": impute_runtime,
                        "n_cells": data.n_cells_used,
                        "n_genes": data.n_features_used,
                        "n_labels": n_labels,
                        "n_batches": n_batches,
                        "label_column": data.label_name,
                        "batch_column": data.batch_name,
                        "feature_mode": data.feature_mode,
                        "n_seeds_requested": len(seeds),
                        "source_note": data.source_note,
                    }
                )
                if status != "ok" or log_imp is None:
                    print(f"    [{status.upper()}] {error}", flush=True)
                    if method == "dca":
                        break
                    continue

                cluster_seeds = seeds if method not in STOCHASTIC_IMPUTERS else [impute_seed]
                for cluster_seed in cluster_seeds:
                    print(f"    clustering seed={cluster_seed}", flush=True)
                    try:
                        rows = cluster_grid(
                            log_imp,
                            data.labels,
                            data.batch,
                            n_pcs_values=n_pcs_values,
                            n_neighbors_values=n_neighbors_values,
                            resolutions=resolutions,
                            seed=cluster_seed,
                            silhouette_sample_size=args.silhouette_sample_size,
                        )
                        for row in rows:
                            row.update(
                                {
                                    "dataset": data.name,
                                    "path": data.path,
                                    "method": method,
                                    "display_method": DISPLAY_NAMES.get(method, method),
                                    "impute_seed": impute_seed,
                                    "cluster_seed": cluster_seed,
                                    "label_column": data.label_name,
                                    "batch_column": data.batch_name,
                                    "n_cells": data.n_cells_used,
                                    "n_genes": data.n_features_used,
                                    "n_labels": n_labels,
                                    "n_batches": n_batches,
                                    "feature_mode": data.feature_mode,
                                }
                            )
                            grid_rows.append(row)
                    except Exception as exc:
                        print(f"    [CLUSTER ERROR] {exc}", flush=True)
                if method == "dca" and method_failed:
                    break

    grid_df = pd.DataFrame(grid_rows)
    runs_df = pd.DataFrame(run_rows)
    summary_df = summarize(grid_df, runs_df)

    grid_path = out_dir / "real_data_clustering_grid.tsv"
    runs_path = out_dir / "real_data_method_runs.tsv"
    summary_path = out_dir / "real_data_clustering_summary.tsv"
    grid_df.to_csv(grid_path, sep="\t", index=False)
    runs_df.to_csv(runs_path, sep="\t", index=False)
    summary_df.to_csv(summary_path, sep="\t", index=False)
    write_tex_summary(summary_df, Path(args.paper_table))
    print(f"Wrote {grid_path}")
    print(f"Wrote {runs_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {args.paper_table}")


if __name__ == "__main__":
    main()
