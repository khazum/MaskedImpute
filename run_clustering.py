#!/usr/bin/env python3
"""
run_clustering.py
----------------

Run clustering evaluation (ARI, NMI, Purity, ASW) on SingleCellExperiment .rds
files for multiple imputation methods. Methods mirror run_imputation.py and
add the experiment autoencoder: magic, dca, autoclass, low_mse, balanced_mse,
experiment.

Procedure (per method):
- impute (or baseline) to obtain logcounts
- PCA to at most 50 components
- k-means with k = number of unique labels (min 2)
- metrics: ARI, NMI, Purity, ASW
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from clustering_eval import evaluate_clustering

try:
    from rds2py import read_rds
except Exception as exc:
    raise SystemExit(
        "Failed to import rds2py. Install requirements or run in the proper env.\n"
        f"Error: {exc}"
    ) from exc

try:
    import run_imputation as imp
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Failed to import run_imputation.py. Ensure it is on PYTHONPATH.\n"
        f"Error: {exc}"
    ) from exc

import torch

METHODS = ("magic", "dca", "autoclass", "low_mse", "balanced_mse", "experiment")


def _counts_obs_from_logcounts(logcounts: np.ndarray, counts: Optional[np.ndarray]) -> np.ndarray:
    if counts is None:
        return np.clip(np.expm1(logcounts * np.log(2.0)), 0.0, None).astype(np.float32)
    return np.clip(counts, 0.0, None).astype(np.float32)


def _cell_zero_norm(zeros_obs: np.ndarray) -> np.ndarray:
    cell_zero_frac = zeros_obs.mean(axis=1).astype(np.float32)
    cz_lo = float(np.percentile(cell_zero_frac, 5.0))
    cz_hi = float(np.percentile(cell_zero_frac, 95.0))
    cz_span = max(cz_hi - cz_lo, imp.EPSILON)
    return np.clip((cell_zero_frac - cz_lo) / cz_span, 0.0, 1.0).astype(np.float32)


def _run_masked26_clustering(
    logcounts: np.ndarray,
    counts: Optional[np.ndarray],
    *,
    bio_reg_weight: float,
    seed: int,
) -> np.ndarray:
    """Run masked_imputation26 without preserving observed nonzeros (clustering only)."""
    mi26 = imp._import_masked26()
    device = torch.device("cuda")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for masked_imputation26 but not available.")

    counts_obs = _counts_obs_from_logcounts(logcounts, counts)
    zeros_obs = counts_obs <= 0.0
    counts_max = counts_obs.max(axis=0)
    cell_zero_norm = _cell_zero_norm(zeros_obs)

    mi26.AE_PARAMS["bio_reg_weight"] = float(bio_reg_weight)
    mi26.set_seed(int(seed))
    p_bio = mi26.splat_cellaware_bio_prob(
        counts=counts_obs,
        zeros_obs=zeros_obs,
        disp_mode=mi26.BIO_PARAMS["disp_mode"],
        use_cell_factor=mi26.BIO_PARAMS["use_cell_factor"],
    )
    if float(mi26.BIO_PARAMS["cell_zero_weight"]) > 0.0:
        cell_w = np.clip(
            float(mi26.BIO_PARAMS["cell_zero_weight"]) * cell_zero_norm, 0.0, 1.0
        )
        p_bio = p_bio * (1.0 - cell_w[:, None])

    log_recon = mi26.train_autoencoder_reconstruct(
        logcounts=logcounts,
        counts_max=counts_max,
        p_bio=p_bio,
        device=device,
        fast_mode=True,
        amp_enabled=True,
        compile_enabled=True,
        fast_batch_mult=2,
        num_workers=2,
    )

    # NOTE: for clustering we do NOT preserve observed nonzeros
    return log_recon.astype(np.float32, copy=False)


LABEL_KEYS = ("cell_type1", "labels", "Group", "label")


def _extract_labels(sce) -> Tuple[np.ndarray, str]:
    colmd = getattr(sce, "column_data", None) or getattr(sce, "colData", None)
    y = None
    source = None
    if colmd is not None:
        if hasattr(colmd, "get_column_names") and hasattr(colmd, "get_column"):
            colnames = list(map(str, colmd.get_column_names()))
            for key in LABEL_KEYS:
                if key in colnames:
                    y = np.asarray(colmd.get_column(key))
                    source = key
                    break
        elif hasattr(colmd, "columns"):
            colnames = list(map(str, getattr(colmd, "columns", [])))
            for key in LABEL_KEYS:
                if key in colnames:
                    y = np.asarray(colmd[key])
                    source = key
                    break
        elif isinstance(colmd, dict):
            for key in LABEL_KEYS:
                if key in colmd:
                    y = np.asarray(colmd[key])
                    source = key
                    break

    if y is None:
        raise RuntimeError(f"No label column found. Tried: {', '.join(LABEL_KEYS)}")

    # factorize labels to integer ids
    uniq, labels = np.unique(np.asarray(y), return_inverse=True)
    labels = labels.astype(int)
    return labels, source or "unknown"


def load_dataset(path: str) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, np.ndarray], np.ndarray, str]:
    sce = read_rds(path)
    if not hasattr(sce, "assay"):
        raise TypeError(f"Unsupported RDS object (expected SingleCellExperiment): {type(sce)}")
    logcounts = sce.assay("logcounts").T.astype("float32")
    counts = None
    try:
        counts = sce.assay("counts").T.astype("float32")
    except Exception:
        counts = None
    try:
        norm = imp.get_normalization_info(sce)
    except Exception:
        norm = {"size_factors": None}
    labels, source = _extract_labels(sce)
    if logcounts.shape[0] != labels.shape[0]:
        raise ValueError(f"Cells mismatch: logcounts {logcounts.shape[0]} vs labels {labels.shape[0]}")
    return logcounts, counts, norm, labels, source


def parse_methods(raw: Optional[str]) -> List[str]:
    if not raw or raw.lower() == "all":
        return list(METHODS)
    methods = [m.strip().lower() for m in raw.split(",") if m.strip()]
    unknown = sorted(set(methods) - set(METHODS))
    if unknown:
        raise ValueError(f"Unknown methods: {', '.join(unknown)}. Allowed: {', '.join(METHODS)} or 'all'.")
    return methods


def _run_method(
    method: str,
    logcounts: np.ndarray,
    counts: Optional[np.ndarray],
    norm: Dict[str, np.ndarray],
    args: argparse.Namespace,
    seed: int,
) -> np.ndarray:
    method = method.lower()
    try:
        if method == "magic":
            return imp.run_magic(logcounts, n_jobs=args.n_jobs)
        if method == "autoclass":
            return imp.run_autoclass(logcounts, args.autoclass_dir, args.autoclass_kwargs)
        if method == "dca":
            if counts is None:
                raise RuntimeError("DCA requires raw counts assay.")
            size_factors = norm.get("size_factors")
            if size_factors is None:
                raise RuntimeError("Missing TrueCounts size factors for DCA normalization.")
            mu = imp.run_dca(
                counts=counts,
                dca_bin=args.dca_bin,
                ae_type=args.dca_type,
                epochs=args.dca_epochs,
                batch_size=args.dca_batch_size,
                threads=args.dca_threads,
                ridge=args.dca_ridge,
                verbose=args.verbose,
            )
            return imp.normalize_counts_to_logcounts(mu, size_factors)
        if method == "low_mse":
            return _run_masked26_clustering(logcounts, counts, bio_reg_weight=0.0, seed=seed)
        if method == "balanced_mse":
            return _run_masked26_clustering(logcounts, counts, bio_reg_weight=1.0, seed=seed)
        if method == "experiment":
            try:
                import experiment as exp
            except Exception as exc:
                raise RuntimeError(f"Failed to import experiment.py: {exc}") from exc
            exp_kwargs = dict(getattr(args, "experiment_kwargs", {}) or {})
            return exp.run_experiment_imputation(logcounts, seed=seed, **exp_kwargs)
    except BaseException as exc:
        raise RuntimeError(str(exc)) from exc
    raise ValueError(f"Unknown method: {method}")


def collect_rds_files(input_path: str) -> List[Path]:
    path = Path(input_path)
    if path.is_dir():
        return sorted(path.rglob("*.rds"))
    if path.is_file():
        return [path]
    raise FileNotFoundError(f"Input path not found: {input_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Clustering evaluation for SCE .rds files")
    parser.add_argument("input_path", help="Input .rds file or directory")
    parser.add_argument("output_dir", help="Output directory for clustering results")
    parser.add_argument(
        "--methods",
        default=None,
        help="Comma-separated list (magic,dca,autoclass,low_mse,balanced_mse,experiment) or 'all'.",
    )
    parser.add_argument(
        "methods_arg",
        nargs="?",
        default=None,
        help="Optional methods list (magic,dca,autoclass,low_mse,balanced_mse,experiment) or 'all'.",
    )
    parser.add_argument("--n-jobs", type=int, default=1, help="MAGIC n_jobs value")
    parser.add_argument("--n-repeat", type=int, default=5, help="Number of repeats per method")

    g_dca = parser.add_argument_group("DCA Options")
    g_dca.add_argument("--dca-bin", default="~/miniconda3/envs/dca_env/bin/dca", help="Path to DCA binary (for DCA method)")
    g_dca.add_argument("--dca-type", default=None, help="DCA --type (e.g., nb-conddisp)")
    g_dca.add_argument("--dca-epochs", type=int, default=None, help="DCA --epochs")
    g_dca.add_argument("--dca-batch-size", type=int, default=None, help="DCA --batch-size")
    g_dca.add_argument("--dca-threads", type=int, default=None, help="DCA --threads")
    g_dca.add_argument("--dca-ridge", type=float, default=None, help="DCA --ridge")

    g_ac = parser.add_argument_group("AutoClass Options")
    g_ac.add_argument("--autoclass-dir", default="AutoClass", help="Path to AutoClass repo (optional)")
    g_ac.add_argument("--autoclass-kwargs", default="", help="Comma-separated key=value overrides")

    g_exp = parser.add_argument_group("Experiment Options")
    g_exp.add_argument(
        "--experiment-kwargs",
        default="",
        help="Comma-separated key=value overrides for experiment method (e.g., epochs=50,bottleneck=16,masked_denoise=true).",
    )

    parser.add_argument("--verbose", action="store_true", help="Verbose method output (e.g., DCA)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = collect_rds_files(args.input_path)
    if not files:
        raise SystemExit("No .rds files found.")

    methods = parse_methods(args.methods_arg or args.methods)
    try:
        args.autoclass_kwargs = imp._parse_kv_pairs(args.autoclass_kwargs)
    except Exception as exc:
        raise SystemExit(f"Invalid --autoclass-kwargs: {exc}") from exc
    try:
        args.experiment_kwargs = imp._parse_kv_pairs(args.experiment_kwargs)
    except Exception as exc:
        raise SystemExit(f"Invalid --experiment-kwargs: {exc}") from exc

    method_rows: Dict[str, List[Dict[str, object]]] = {m: [] for m in methods}

    for rds_path in files:
        name = rds_path.stem
        print(f"\n--- Dataset: {name} ---")
        try:
            logcounts, counts, norm, labels, label_source = load_dataset(str(rds_path))
        except Exception as exc:
            print(f"  [ERROR] {name}: {exc}")
            continue

        for method in methods:
            print(f"  -> {method}")
            metrics_list: List[Dict[str, float]] = []
            runtimes: List[float] = []
            err_msg: Optional[str] = None
            for rep in range(args.n_repeat):
                seed = 42 + rep
                t0 = time.time()
                try:
                    log_imp = _run_method(method, logcounts, counts, norm, args, seed)
                    res = evaluate_clustering(log_imp, labels)
                    metrics_list.append(res)
                except Exception as exc:
                    err_msg = str(exc)
                    print(f"    [ERROR] {method}: {exc}")
                    break
                runtimes.append(time.time() - t0)
            if metrics_list:
                def _mean_std(key: str) -> Tuple[float, float]:
                    vals = [m[key] for m in metrics_list]
                    return float(np.nanmean(vals)), float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0

                asw_m, asw_s = _mean_std("ASW")
                ari_m, ari_s = _mean_std("ARI")
                nmi_m, nmi_s = _mean_std("NMI")
                ps_m, ps_s = _mean_std("PS")
                rt_m = float(np.mean(runtimes)) if runtimes else float("nan")
                rt_s = float(np.std(runtimes, ddof=1)) if len(runtimes) > 1 else 0.0
                row = {
                    "dataset": name,
                    "ASW": asw_m,
                    "ASW_std": asw_s,
                    "ARI": ari_m,
                    "ARI_std": ari_s,
                    "NMI": nmi_m,
                    "NMI_std": nmi_s,
                    "PS": ps_m,
                    "PS_std": ps_s,
                    "runtime_sec": rt_m,
                    "runtime_sec_std": rt_s,
                    "n_repeats": len(metrics_list),
                    "n_cells": int(logcounts.shape[0]),
                    "n_genes": int(logcounts.shape[1]),
                    "label_source": label_source,
                    "error": err_msg or "",
                }
            else:
                row = {
                    "dataset": name,
                    "ASW": float("nan"),
                    "ASW_std": float("nan"),
                    "ARI": float("nan"),
                    "ARI_std": float("nan"),
                    "NMI": float("nan"),
                    "NMI_std": float("nan"),
                    "PS": float("nan"),
                    "PS_std": float("nan"),
                    "runtime_sec": float("nan"),
                    "runtime_sec_std": float("nan"),
                    "n_repeats": 0,
                    "n_cells": int(logcounts.shape[0]),
                    "n_genes": int(logcounts.shape[1]),
                    "label_source": label_source,
                    "error": err_msg or "method failed",
                }
            method_rows[method].append(row)
            if metrics_list:
                print(
                    f"    ASW={row['ASW']:.4f} ARI={row['ARI']:.4f} NMI={row['NMI']:.4f} PS={row['PS']:.4f}"
                )

    # save per-method tables
    import csv

    columns = [
        "dataset",
        "ASW",
        "ASW_std",
        "ARI",
        "ARI_std",
        "NMI",
        "NMI_std",
        "PS",
        "PS_std",
        "runtime_sec",
        "runtime_sec_std",
        "n_repeats",
        "n_cells",
        "n_genes",
        "label_source",
        "error",
    ]
    for method, rows in method_rows.items():
        out_path = out_dir / f"{method}_clustering_table.tsv"
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns, delimiter="\t")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        print(f"Saved {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
