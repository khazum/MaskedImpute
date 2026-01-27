#!/usr/bin/env python3
"""
splat_predict.py
----------------

Predict biological zeros among observed zeros using the SPLAT cell-aware
posterior, then zero those entries in logTrueCounts and report MSE against
logTrueCounts. The decision threshold is tuned globally across all datasets.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from rds2py import read_rds

from predict_dropouts_new import splatter_bio_posterior_from_counts, _choose_thresh_for_metric

EPSILON = 1e-6


def _masked_mse(diff: np.ndarray, mask: np.ndarray) -> float:
    n = int(mask.sum())
    if n == 0:
        return float("nan")
    return float(np.mean((diff[mask]) ** 2))


def compute_mse_metrics(log_imp: np.ndarray, log_true: np.ndarray, log_obs: np.ndarray) -> Dict[str, float]:
    diff = log_true - log_imp
    mask_biozero = log_true <= EPSILON
    mask_dropout = (log_true > EPSILON) & (log_obs <= EPSILON)
    mask_non_zero = (log_true > EPSILON) & (log_obs > EPSILON)
    return {
        "mse": float(np.mean(diff ** 2)),
        "mse_dropout": _masked_mse(diff, mask_dropout),
        "mse_biozero": _masked_mse(diff, mask_biozero),
        "mse_non_zero": _masked_mse(diff, mask_non_zero),
        "n_total": int(diff.size),
        "n_dropout": int(mask_dropout.sum()),
        "n_biozero": int(mask_biozero.sum()),
        "n_non_zero": int(mask_non_zero.sum()),
    }


def logcounts_to_counts(logcts: np.ndarray, base: float = 2.0) -> np.ndarray:
    return np.expm1(logcts * np.log(base))


def _sanitize_prob(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(p, 0.0, 1.0)


def load_dataset(path: str) -> Optional[Dict[str, np.ndarray]]:
    sce = read_rds(path)
    if not hasattr(sce, "assay"):
        raise TypeError(f"Unsupported RDS object (expected SingleCellExperiment): {type(sce)}")

    logcounts = sce.assay("logcounts").T.astype("float32")
    keep = np.sum(logcounts != 0, axis=0) >= 2
    logcounts = logcounts[:, keep]

    log_true = None
    for assay_name in ("logTrueCounts", "perfect_logcounts"):
        try:
            log_true = sce.assay(assay_name).T[:, keep].astype("float32")
            break
        except Exception:
            continue

    if log_true is None:
        return None

    return {"logcounts": logcounts, "log_true": log_true}


def collect_rds_files(input_path: str) -> List[Path]:
    path = Path(input_path)
    if path.is_dir():
        return sorted(path.rglob("*.rds"))
    if path.is_file():
        return [path]
    raise FileNotFoundError(f"Input path not found: {input_path}")


def splat_cellaware_p_bio_at_zeros(counts: np.ndarray, zeros_obs: np.ndarray) -> np.ndarray:
    bio_post = splatter_bio_posterior_from_counts(
        counts,
        disp_mode="estimate",
        disp_const=0.1,
        use_cell_factor=True,
        groups=None,
    )
    return _sanitize_prob(np.asarray(bio_post, dtype=np.float64)[zeros_obs]).astype(np.float32)


def choose_global_threshold(
    p_bio_list: List[np.ndarray],
    bio_true_list: List[np.ndarray],
    drop_true_list: List[np.ndarray],
    metric: str,
) -> float:
    if not p_bio_list:
        return float("nan")
    p_drop_all = np.concatenate([(1.0 - _sanitize_prob(p)).reshape(-1) for p in p_bio_list], axis=0)
    bio_all = np.concatenate([np.asarray(b, dtype=bool).reshape(-1) for b in bio_true_list], axis=0)
    drop_all = np.concatenate([np.asarray(d, dtype=bool).reshape(-1) for d in drop_true_list], axis=0)
    zeros_all = np.ones_like(bio_all, dtype=bool)
    return float(_choose_thresh_for_metric(p_drop_all, zeros_all, drop_all, bio_all, metric=metric))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict biological zeros using SPLAT cell-aware posterior and report MSE."
    )
    parser.add_argument("input_path", help="Path to .rds file or directory")
    parser.add_argument("output_dir", help="Output directory for splat_cellaware_mse_table.tsv")
    parser.add_argument(
        "--metric",
        default="f1",
        choices=("precision", "recall", "accuracy", "f1"),
        help="Metric used to tune the global threshold.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets: List[Dict[str, object]] = []
    p_bio_list: List[np.ndarray] = []
    bio_true_list: List[np.ndarray] = []
    drop_true_list: List[np.ndarray] = []

    for path in collect_rds_files(args.input_path):
        ds_name = path.stem
        data = load_dataset(str(path))
        if data is None:
            print(f"[WARN] {ds_name}: missing logTrueCounts; skipping.")
            continue

        logcounts = data["logcounts"]
        log_true = data["log_true"]

        counts_obs = np.clip(logcounts_to_counts(logcounts), 0.0, None)
        counts_true = np.clip(logcounts_to_counts(log_true), 0.0, None)
        zeros_obs = counts_obs <= 0.0

        p_bio_z = splat_cellaware_p_bio_at_zeros(counts_obs, zeros_obs)
        bio_true_z = (counts_true <= 0.0)[zeros_obs].astype(bool)
        drop_true_z = (counts_true > 0.0)[zeros_obs].astype(bool)

        p_bio_list.append(p_bio_z)
        bio_true_list.append(bio_true_z)
        drop_true_list.append(drop_true_z)

        datasets.append(
            {
                "dataset": ds_name,
                "logcounts": logcounts,
                "log_true": log_true,
                "zeros_obs": zeros_obs,
                "p_bio_z": p_bio_z,
            }
        )

    if not datasets:
        raise SystemExit("No datasets processed.")

    thr_drop = choose_global_threshold(p_bio_list, bio_true_list, drop_true_list, args.metric)
    thr_bio = float("nan") if not np.isfinite(thr_drop) else 1.0 - float(thr_drop)
    print(f"Global threshold (metric={args.metric}): P(dropout) < {thr_drop:.4f} (P(bio) >= {thr_bio:.4f})")

    rows: List[Dict[str, object]] = []
    for ds in datasets:
        ds_name = str(ds["dataset"])
        logcounts = np.asarray(ds["logcounts"], dtype=np.float32)
        log_true = np.asarray(ds["log_true"], dtype=np.float32)
        zeros_obs = np.asarray(ds["zeros_obs"], dtype=bool)
        p_bio_z = np.asarray(ds["p_bio_z"], dtype=np.float32)

        if np.isfinite(thr_drop):
            pred_bio_z = (1.0 - p_bio_z) < thr_drop
        else:
            pred_bio_z = np.zeros_like(p_bio_z, dtype=bool)

        pred_mask = np.zeros_like(zeros_obs, dtype=bool)
        pred_mask[zeros_obs] = pred_bio_z

        counts_true = np.clip(logcounts_to_counts(log_true), 0.0, None)
        bio_true_z = (counts_true <= 0.0)[zeros_obs].astype(bool)
        tp = int((pred_bio_z & bio_true_z).sum())
        fn = int((~pred_bio_z & bio_true_z).sum())
        recall_bio = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")

        log_adj = log_true.copy()
        log_adj[zeros_obs] = 1.0
        log_adj[pred_mask] = 0.0

        row = {
            "dataset": ds_name,
            "thr_drop": float(thr_drop),
            "thr_bio": float(thr_bio),
            "recall_bio": recall_bio,
            "n_obs_zero": int(zeros_obs.sum()),
            "n_pred_bio": int(pred_bio_z.sum()),
            **compute_mse_metrics(log_adj, log_true, logcounts),
        }
        rows.append(row)

    out_path = output_dir / "splat_cellaware_mse_table.tsv"
    columns = [
        "dataset",
        "thr_drop",
        "thr_bio",
        "recall_bio",
        "mse",
        "mse_dropout",
        "mse_biozero",
        "mse_non_zero",
        "n_total",
        "n_dropout",
        "n_biozero",
        "n_non_zero",
        "n_obs_zero",
        "n_pred_bio",
    ]
    with out_path.open("w", encoding="utf-8") as f:
        f.write("\t".join(columns) + "\n")
        for row in sorted(rows, key=lambda r: str(r["dataset"])):
            f.write("\t".join(str(row.get(col, "")) for col in columns) + "\n")

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
