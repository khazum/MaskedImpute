#!/usr/bin/env python3
"""
splat_predict2.py
-----------------

Improved SPLAT cell-aware bio-zero prediction:
  - Uses raw counts when available (assay "counts")
  - Optionally stratifies by Group/Batch (auto-selects best mode)
  - Picks a single global threshold that maximizes F-beta (beta>1 favors recall)

Predicted biological zeros among observed zeros are set to 0 in logTrueCounts
after setting all observed zeros to 1, then MSE is computed vs logTrueCounts.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from rds2py import read_rds

from predict_dropouts_new import splatter_bio_posterior_from_counts

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


def _extract_coldata(sce) -> Dict[str, np.ndarray]:
    colmd = getattr(sce, "column_data", None) or getattr(sce, "colData", None)
    if colmd is None and isinstance(sce, dict):
        colmd = sce.get("colData") or sce.get("column_data")

    extracted_cols: Dict[str, np.ndarray] = {}
    if colmd is not None:
        if hasattr(colmd, "get_column_names") and hasattr(colmd, "get_column"):
            try:
                colnames = list(map(str, colmd.get_column_names()))
                for name in colnames:
                    extracted_cols[name] = np.asarray(colmd.get_column(name))
            except Exception:
                pass
        elif hasattr(colmd, "columns"):
            try:
                colnames = list(map(str, getattr(colmd, "columns", [])))
                for name in colnames:
                    if hasattr(colmd, "__getitem__"):
                        extracted_cols[name] = np.asarray(colmd[name])
            except Exception:
                pass
        elif isinstance(colmd, dict):
            target_dict = colmd.get("listData", colmd)
            for k, v in target_dict.items():
                if hasattr(v, "__len__"):
                    extracted_cols[k] = np.asarray(v)
    return extracted_cols


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

    counts = None
    try:
        counts = sce.assay("counts").T.astype("float32")
        counts = counts[:, keep]
    except Exception:
        counts = None

    coldata = _extract_coldata(sce)
    group = None
    batch = None
    for key in ("Group", "group"):
        if key in coldata:
            group = coldata[key].astype(str)
            break
    for key in ("Batch", "batch"):
        if key in coldata:
            batch = coldata[key].astype(str)
            break

    return {
        "logcounts": logcounts,
        "log_true": log_true,
        "counts": counts,
        "group": group,
        "batch": batch,
    }


def collect_rds_files(input_path: str) -> List[Path]:
    path = Path(input_path)
    if path.is_dir():
        return sorted(path.rglob("*.rds"))
    if path.is_file():
        return [path]
    raise FileNotFoundError(f"Input path not found: {input_path}")


def _build_groups(mode: str, group: Optional[np.ndarray], batch: Optional[np.ndarray]) -> Optional[np.ndarray]:
    mode = mode.lower()
    if mode == "none":
        return None
    if mode == "group":
        if group is None:
            return None
        return group
    if mode == "batch":
        if batch is None:
            return None
        return batch
    if mode == "group_batch":
        if group is None and batch is None:
            return None
        if group is None:
            return batch
        if batch is None:
            return group
        return np.asarray([f"{g}|{b}" for g, b in zip(group, batch)], dtype=object)
    raise ValueError(f"Unknown group mode: {mode}")


def splat_cellaware_p_bio_at_zeros(
    counts: np.ndarray,
    zeros_obs: np.ndarray,
    groups: Optional[np.ndarray],
) -> np.ndarray:
    bio_post = splatter_bio_posterior_from_counts(
        counts,
        disp_mode="estimate",
        disp_const=0.1,
        use_cell_factor=True,
        groups=groups,
    )
    return _sanitize_prob(np.asarray(bio_post, dtype=np.float64)[zeros_obs]).astype(np.float32)


def choose_threshold_max_fbeta(
    p_drop: np.ndarray,
    bio_true: np.ndarray,
    drop_true: np.ndarray,
    grid: np.ndarray,
    beta: float,
    eps: float = 1e-12,
) -> Tuple[float, float, float, float]:
    best_t = float("nan")
    best_fbeta = -1.0
    best_rec = -1.0
    best_prec = -1.0
    beta2 = float(beta) ** 2

    for t in grid:
        pred_bio = p_drop < t
        tp = int((pred_bio & bio_true).sum())
        fn = int((~pred_bio & bio_true).sum())
        fp = int((pred_bio & drop_true).sum())

        rec = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        denom = (beta2 * prec) + rec
        fbeta = (1.0 + beta2) * prec * rec / denom if denom > 0 else 0.0

        if (fbeta > best_fbeta + eps) or (abs(fbeta - best_fbeta) <= eps and rec > best_rec + eps):
            best_fbeta = float(fbeta)
            best_rec = float(rec)
            best_prec = float(prec)
            best_t = float(t)

    return best_t, best_fbeta, best_rec, best_prec


def global_metrics(p_bio_list: List[np.ndarray], bio_true_list: List[np.ndarray], drop_true_list: List[np.ndarray], thr_drop: float, beta: float) -> Dict[str, float]:
    if not p_bio_list:
        return {"fbeta": float("nan"), "f1": float("nan"), "recall": float("nan"), "precision": float("nan")}
    p_drop = np.concatenate([(1.0 - _sanitize_prob(p)).reshape(-1) for p in p_bio_list], axis=0)
    bio_true = np.concatenate([np.asarray(b, dtype=bool).reshape(-1) for b in bio_true_list], axis=0)
    drop_true = np.concatenate([np.asarray(d, dtype=bool).reshape(-1) for d in drop_true_list], axis=0)

    pred_bio = p_drop < float(thr_drop)
    tp = int((pred_bio & bio_true).sum())
    fn = int((~pred_bio & bio_true).sum())
    fp = int((pred_bio & drop_true).sum())

    rec = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2.0 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    beta2 = float(beta) ** 2
    denom = (beta2 * prec) + rec
    fbeta = (1.0 + beta2) * prec * rec / denom if denom > 0 else 0.0
    return {"fbeta": float(fbeta), "f1": float(f1), "recall": float(rec), "precision": float(prec)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Improved SPLAT cell-aware bio-zero prediction with global F1+recall thresholding."
    )
    parser.add_argument("input_path", help="Path to .rds file or directory")
    parser.add_argument("output_dir", help="Output directory for splat_cellaware2_mse_table.tsv")
    parser.add_argument(
        "--grid",
        type=int,
        default=99,
        help="Number of threshold grid points in [0.01, 0.99] (default: 99).",
    )
    parser.add_argument(
        "--group-mode",
        choices=("auto", "none", "group", "batch", "group_batch"),
        default="auto",
        help="Ignored; all modes are evaluated and the best score is selected.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=2.0,
        help="F-beta weight for threshold selection (beta>1 favors recall).",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets: List[Dict[str, object]] = []
    for path in collect_rds_files(args.input_path):
        ds_name = path.stem
        data = load_dataset(str(path))
        if data is None:
            print(f"[WARN] {ds_name}: missing logTrueCounts; skipping.")
            continue

        logcounts = data["logcounts"]
        log_true = data["log_true"]
        counts = data["counts"]
        group = data["group"]
        batch = data["batch"]

        if counts is None:
            counts = np.clip(logcounts_to_counts(logcounts), 0.0, None)
        else:
            counts = np.clip(counts, 0.0, None)

        zeros_obs = counts <= 0.0
        bio_true_z = (log_true <= EPSILON)[zeros_obs].astype(bool)
        drop_true_z = (log_true > EPSILON)[zeros_obs].astype(bool)

        datasets.append(
            {
                "dataset": ds_name,
                "logcounts": logcounts,
                "log_true": log_true,
                "counts": counts,
                "zeros_obs": zeros_obs,
                "bio_true_z": bio_true_z,
                "drop_true_z": drop_true_z,
                "group": group,
                "batch": batch,
            }
        )

    if not datasets:
        raise SystemExit("No datasets processed.")

    grid = np.linspace(0.01, 0.99, int(args.grid))

    modes = ["none", "group", "batch", "group_batch"]
    if hasattr(args, "group_mode") and args.group_mode != "auto":
        print(f"[WARN] --group-mode={args.group_mode} ignored; selecting best mode by F-beta.")
    chosen_mode = "auto"
    mode_stats: Dict[str, Dict[str, float]] = {}
    mode_thr: Dict[str, float] = {}
    mode_pbio: Dict[str, List[np.ndarray]] = {}

    for mode in modes:
        p_bio_list: List[np.ndarray] = []
        for ds in datasets:
            groups = _build_groups(mode, ds["group"], ds["batch"])
            if groups is not None:
                unique = np.unique(groups)
                if unique.size <= 1:
                    groups = None
            p_bio_z = splat_cellaware_p_bio_at_zeros(ds["counts"], ds["zeros_obs"], groups)
            p_bio_list.append(p_bio_z)

        thr_drop, best_fbeta, best_rec, best_prec = choose_threshold_max_fbeta(
            p_drop=np.concatenate([(1.0 - _sanitize_prob(p)).reshape(-1) for p in p_bio_list], axis=0),
            bio_true=np.concatenate([ds["bio_true_z"] for ds in datasets], axis=0),
            drop_true=np.concatenate([ds["drop_true_z"] for ds in datasets], axis=0),
            grid=grid,
            beta=float(args.beta),
        )

        mode_thr[mode] = float(thr_drop)
        mode_stats[mode] = {"fbeta": float(best_fbeta), "recall": float(best_rec), "precision": float(best_prec)}
        mode_pbio[mode] = p_bio_list

    best_mode = None
    best_fbeta = -1.0
    best_rec = -1.0
    for mode in modes:
        stats = mode_stats[mode]
        fbeta = stats["fbeta"]
        rec = stats["recall"]
        if (fbeta > best_fbeta) or (fbeta == best_fbeta and rec > best_rec):
            best_mode = mode
            best_fbeta = fbeta
            best_rec = rec
    chosen_mode = best_mode or "none"

    thr_drop = float(mode_thr[chosen_mode])
    thr_bio = 1.0 - thr_drop if np.isfinite(thr_drop) else float("nan")
    print(
        f"Chosen mode: {chosen_mode} | thr_drop={thr_drop:.4f} | "
        f"Fbeta={mode_stats[chosen_mode]['fbeta']:.4f} | Recall={mode_stats[chosen_mode]['recall']:.4f}"
    )

    rows: List[Dict[str, object]] = []
    for ds, p_bio_z in zip(datasets, mode_pbio[chosen_mode]):
        ds_name = str(ds["dataset"])
        logcounts = np.asarray(ds["logcounts"], dtype=np.float32)
        log_true = np.asarray(ds["log_true"], dtype=np.float32)
        zeros_obs = np.asarray(ds["zeros_obs"], dtype=bool)
        bio_true_z = np.asarray(ds["bio_true_z"], dtype=bool)
        drop_true_z = np.asarray(ds["drop_true_z"], dtype=bool)

        if np.isfinite(thr_drop):
            pred_bio_z = (1.0 - p_bio_z) < thr_drop
        else:
            pred_bio_z = np.zeros_like(p_bio_z, dtype=bool)

        pred_mask = np.zeros_like(zeros_obs, dtype=bool)
        pred_mask[zeros_obs] = pred_bio_z

        tp = int((pred_bio_z & bio_true_z).sum())
        fn = int((~pred_bio_z & bio_true_z).sum())
        fp = int((pred_bio_z & drop_true_z).sum())
        rec = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2.0 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        beta2 = float(args.beta) ** 2
        denom = (beta2 * prec) + rec
        fbeta = (1.0 + beta2) * prec * rec / denom if denom > 0 else 0.0

        log_adj = log_true.copy()
        log_adj[zeros_obs] = 1.0
        log_adj[pred_mask] = 0.0

        row = {
            "dataset": ds_name,
            "group_mode": chosen_mode,
            "thr_drop": float(thr_drop),
            "thr_bio": float(thr_bio),
            "beta": float(args.beta),
            "precision_bio": float(prec),
            "recall_bio": float(rec),
            "f1_bio": float(f1),
            "fbeta_bio": float(fbeta),
            "n_obs_zero": int(zeros_obs.sum()),
            "n_pred_bio": int(pred_bio_z.sum()),
            **compute_mse_metrics(log_adj, log_true, logcounts),
        }
        rows.append(row)

    out_path = output_dir / "splat_cellaware2_mse_table.tsv"
    columns = [
        "dataset",
        "group_mode",
        "thr_drop",
        "thr_bio",
        "beta",
        "precision_bio",
        "recall_bio",
        "f1_bio",
        "fbeta_bio",
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
