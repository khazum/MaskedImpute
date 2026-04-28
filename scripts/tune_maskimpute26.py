#!/usr/bin/env python3
"""Evaluate one MaskImpute configuration on synthetic benchmark datasets."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import masked_imputation26 as mi26
import run_imputation as bench


METRIC_COLUMNS = [
    "mse",
    "mse_dropout",
    "mse_biozero",
    "mse_non_zero",
    "mse_marker",
    "mae",
    "mae_dropout",
    "mae_biozero",
    "mae_non_zero",
    "mae_marker",
    "gnrmse",
    "gnrmse_marker",
    "corr_err",
    "runtime_sec",
]


def _finite_mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else float("nan")


def _finite_std(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")


def _load_config(raw: str) -> Dict[str, object]:
    path = Path(raw)
    if path.exists():
        return json.loads(path.read_text())
    return json.loads(raw)


def _apply_config(config: Dict[str, object]) -> None:
    for key, value in config.items():
        if key in mi26.AE_PARAMS:
            mi26.AE_PARAMS[key] = value
        elif key in mi26.MODEL_PARAMS:
            mi26.MODEL_PARAMS[key] = value
        elif key == "zero_shrink_strength":
            mi26.ZERO_SHRINK_STRENGTH = float(value)
        elif key == "observed_recon_weight":
            mi26.OBSERVED_RECON_WEIGHT = float(value)
        elif key == "zero_mask_uses_bio_prob":
            mi26.AE_PARAMS[key] = bool(value)
        elif key in {"name", "notes"}:
            continue
        else:
            raise ValueError(f"Unknown MaskImpute tuning key: {key}")


def _counts_obs_from_dataset(logcounts: np.ndarray, counts: Optional[np.ndarray]) -> np.ndarray:
    if counts is None:
        return np.clip(np.expm1(logcounts * np.log(2.0)), 0.0, None).astype(np.float32)
    return np.clip(np.asarray(counts, dtype=np.float32), 0.0, None)


def _collect_files(input_dir: Path, datasets: Optional[List[str]]) -> List[Path]:
    files = sorted(input_dir.rglob("*.rds"))
    if datasets:
        wanted = set(datasets)
        files = [p for p in files if bench.dataset_name_from_path(p) in wanted]
    if not files:
        raise FileNotFoundError(f"No .rds files found under {input_dir}")
    return files


def evaluate_config(
    *,
    input_dir: Path,
    output_dir: Path,
    config: Dict[str, object],
    repeats: int,
    seed: int,
    datasets: Optional[List[str]],
    amp: bool,
    compile_model: bool,
    fast_batch_mult: int,
    num_workers: int,
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    _apply_config(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    files = _collect_files(input_dir, datasets)
    per_dataset: List[Dict[str, object]] = []

    for path in files:
        ds_name = bench.dataset_name_from_path(path)
        dataset = bench.load_dataset(str(path))
        if dataset is None:
            raise RuntimeError(f"{ds_name}: missing logTrueCounts; cannot tune.")

        logcounts = np.asarray(dataset["logcounts"], dtype=np.float32)
        log_true = np.asarray(dataset["log_true"], dtype=np.float32)
        counts_obs = _counts_obs_from_dataset(logcounts, dataset.get("counts"))
        zeros_obs = counts_obs <= 0.0
        counts_max = counts_obs.max(axis=0)
        masks = bench.compute_masks(
            log_true,
            logcounts,
            marker_gene_mask=np.asarray(dataset["marker_gene_mask"], dtype=bool),
        )
        p_bio = mi26.splat_cellaware_bio_prob(
            counts=counts_obs,
            zeros_obs=zeros_obs,
            disp_mode=mi26.BIO_PARAMS["disp_mode"],
            use_cell_factor=mi26.BIO_PARAMS["use_cell_factor"],
        )

        runs: List[Dict[str, float]] = []
        for rep in range(int(repeats)):
            mi26.set_seed(int(seed) + rep)
            t0 = time.perf_counter()
            recon = mi26.train_autoencoder_reconstruct(
                logcounts=logcounts,
                counts_max=counts_max,
                p_bio=p_bio,
                device=device,
                fast_mode=True,
                amp_enabled=bool(amp),
                compile_enabled=bool(compile_model),
                fast_batch_mult=int(fast_batch_mult),
                num_workers=int(num_workers),
            )
            p_refined = mi26.refine_bio_prob_with_reconstruction(
                recon_log=recon,
                counts_obs=counts_obs,
                zeros_obs=zeros_obs,
            )
            log_imp = mi26.apply_zero_gate(recon, p_refined, zeros_obs)
            metrics = bench.compute_error_metrics(log_imp, log_true, masks)
            metrics["runtime_sec"] = float(time.perf_counter() - t0)
            runs.append(metrics)

        row: Dict[str, object] = {
            "dataset": ds_name,
            "n_repeats": int(repeats),
        }
        for metric in METRIC_COLUMNS:
            vals = [float(r.get(metric, float("nan"))) for r in runs]
            row[metric] = _finite_mean(vals)
            row[f"{metric}_std"] = _finite_std(vals)
        row.update(bench.compute_mask_counts(masks))
        per_dataset.append(row)

    summary: Dict[str, object] = {
        "config": config,
        "input_dir": str(input_dir),
        "device": str(device),
        "n_datasets": len(per_dataset),
        "n_repeats": int(repeats),
    }
    for metric in METRIC_COLUMNS:
        summary[metric] = _finite_mean(float(r[metric]) for r in per_dataset)
        summary[f"{metric}_dataset_std"] = _finite_std(float(r[metric]) for r in per_dataset)

    # Selection aids only; final claims should report individual metrics.
    summary["score_mse_plus_biozero"] = float(summary["mse"]) + float(summary["mse_biozero"])
    summary["score_mse_plus_2biozero"] = float(summary["mse"]) + 2.0 * float(summary["mse_biozero"])

    with (output_dir / "per_dataset.tsv").open("w", newline="") as fh:
        fieldnames = ["dataset", "n_repeats"] + [
            col for m in METRIC_COLUMNS for col in (m, f"{m}_std")
        ] + ["n_total", "n_dropout", "n_biozero", "n_non_zero", "n_marker", "n_marker_genes"]
        writer = csv.DictWriter(fh, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_dataset)

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    with (output_dir / "summary.tsv").open("w", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(["metric", "value"])
        for key in sorted(summary):
            if key == "config":
                writer.writerow([key, json.dumps(summary[key], sort_keys=True)])
            else:
                writer.writerow([key, summary[key]])

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", required=True, help="JSON string or path to JSON config.")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--datasets", default="", help="Optional comma-separated dataset names.")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--fast-batch-mult", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()] or None
    summary = evaluate_config(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        config=_load_config(args.config),
        repeats=args.repeats,
        seed=args.seed,
        datasets=datasets,
        amp=not args.no_amp,
        compile_model=bool(args.compile),
        fast_batch_mult=args.fast_batch_mult,
        num_workers=args.num_workers,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
