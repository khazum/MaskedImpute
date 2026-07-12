#!/usr/bin/env python3
"""Screen MaskImpute configurations/post-processing for real-data clustering."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import masked_imputation26 as mi26
import run_imputation as ri

_phase5_path = REPO_ROOT / "scripts" / "run_phase5_real_data.py"
_spec = importlib.util.spec_from_file_location("phase5", _phase5_path)
phase5 = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
sys.modules["phase5"] = phase5
_spec.loader.exec_module(phase5)


def load_json(raw: str) -> Dict[str, object]:
    p = Path(raw)
    if p.exists():
        return json.loads(p.read_text())
    return json.loads(raw)


def apply_config(config: Dict[str, object]) -> None:
    for key, value in config.items():
        if key in mi26.AE_PARAMS:
            mi26.AE_PARAMS[key] = value
        elif key in mi26.MODEL_PARAMS:
            mi26.MODEL_PARAMS[key] = value
        elif key == "zero_shrink_strength":
            mi26.ZERO_SHRINK_STRENGTH = float(value)
        elif key == "observed_recon_weight":
            mi26.OBSERVED_RECON_WEIGHT = float(value)
        elif key in {"name", "notes"}:
            continue
        else:
            raise ValueError(f"Unknown config key: {key}")


def transform_matrix(name: str, imp: np.ndarray, obs: np.ndarray, counts: np.ndarray) -> np.ndarray:
    if name == "raw":
        return imp.astype(np.float32, copy=False)
    if name.startswith("all_obs"):
        w = float(name.replace("all_obs", ""))
        return ((1.0 - w) * imp + w * obs).astype(np.float32)
    if name.startswith("zero_obs"):
        w = float(name.replace("zero_obs", ""))
        out = imp.copy()
        z = counts <= 0.0
        out[z] = (1.0 - w) * out[z]
        return out.astype(np.float32)
    if name.startswith("nonzero_obs"):
        w = float(name.replace("nonzero_obs", ""))
        out = imp.copy()
        nz = counts > 0.0
        out[nz] = (1.0 - w) * out[nz] + w * obs[nz]
        return out.astype(np.float32)
    if name == "row_center":
        out = imp - imp.mean(axis=1, keepdims=True)
        return out.astype(np.float32)
    if name == "row_zscore":
        sd = imp.std(axis=1, keepdims=True)
        out = (imp - imp.mean(axis=1, keepdims=True)) / np.maximum(sd, 1e-6)
        return out.astype(np.float32)
    raise ValueError(f"Unknown transform {name}")


def summarize_grid(rows: List[Dict[str, object]], n_labels: int) -> Dict[str, float]:
    df = pd.DataFrame(rows)
    out = {
        "ari_grid_mean": float(df["ari"].mean()),
        "nmi_grid_mean": float(df["nmi"].mean()),
        "ari_grid_max": float(df["ari"].max()),
        "nmi_grid_max": float(df["nmi"].max()),
        "n_clusters_grid_mean": float(df["n_clusters"].mean()),
    }
    matched = []
    for seed, sub in df.groupby("seed"):
        tmp = sub.copy()
        tmp["cluster_delta"] = (tmp["n_clusters"] - n_labels).abs()
        tmp["res_delta"] = (tmp["resolution"] - 1.0).abs()
        tmp = tmp.sort_values(["cluster_delta", "res_delta", "n_pcs", "n_neighbors"])
        matched.append(tmp.iloc[0])
    m = pd.DataFrame(matched)
    out.update(
        {
            "ari_matched_mean": float(m["ari"].mean()),
            "nmi_matched_mean": float(m["nmi"].mean()),
            "n_clusters_matched_mean": float(m["n_clusters"].mean()),
        }
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="PBMC68k=temp/pbmc68k_data.rds")
    parser.add_argument("--out-dir", type=Path, default=Path("results_real_data/tuning_maskimpute"))
    parser.add_argument("--config", action="append", default=[])
    parser.add_argument("--transforms", default="raw,all_obs0.1,all_obs0.25,all_obs0.5,zero_obs0.25,zero_obs0.5,nonzero_obs0.25,row_center,row_zscore")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cluster-seeds", default="42")
    parser.add_argument("--n-hvg", type=int, default=1000)
    parser.add_argument("--pbmc-max-cells", type=int, default=12000)
    parser.add_argument("--max-cells", type=int, default=0)
    parser.add_argument("--n-pcs-grid", default="20,50,100")
    parser.add_argument("--n-neighbors-grid", default="10,15,30")
    parser.add_argument("--resolution-grid", default="0.2,0.5,1.0,1.5,2.0")
    parser.add_argument("--cache-dir", default="results_real_data/cache")
    parser.add_argument("--reuse-npz", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    configs = [load_json(x) for x in args.config] or [{"name": "default"}]
    transforms = [x.strip() for x in args.transforms.split(",") if x.strip()]
    cluster_seeds = phase5.parse_int_list(args.cluster_seeds)
    n_pcs_values = phase5.parse_int_list(args.n_pcs_grid)
    n_neighbors_values = phase5.parse_int_list(args.n_neighbors_grid)
    resolutions = phase5.parse_float_list(args.resolution_grid)

    # Build a lightweight args namespace compatible with phase5.load_dataset.
    ds_args = argparse.Namespace(
        label_columns="cell_type1,cell_type,CellType,label,labels,clust_id",
        batch_columns="human,donor,batch,sample,Sample,Batch,orig.ident",
        n_hvg=args.n_hvg,
        min_hvg_cells=10,
        max_cells=args.max_cells,
        pbmc_max_cells=args.pbmc_max_cells,
        cell_sample_strategy="random",
        seed=args.seed,
        target_sum=10000.0,
        pbmc_annotations="temp/single-cell-3prime-paper/pbmc68k_analysis/68k_pbmc_barcodes_annotation.tsv",
        cache_dir=args.cache_dir,
        rscript_bin="~/miniconda3/envs/r45_bio/bin/Rscript",
    )
    data = phase5.load_dataset(phase5.parse_dataset_arg(args.dataset), ds_args)
    n_labels = len(set(data.labels.tolist()))
    rows = []
    for cfg_i, config in enumerate(configs):
        name = str(config.get("name", f"cfg{cfg_i}"))
        apply_config(config)
        mi26.set_seed(args.seed)
        npz_path = args.out_dir / f"{data.name}_{name}_seed{args.seed}.npz"
        if args.reuse_npz and npz_path.exists():
            imp = np.load(npz_path)["imp"]
            runtime = float(np.load(npz_path)["runtime"])
        else:
            t0 = time.perf_counter()
            imp = ri.run_maskimpute(data.logcounts, data.counts, seed=args.seed)
            runtime = time.perf_counter() - t0
            np.savez_compressed(npz_path, imp=imp, runtime=np.asarray(runtime))
        for transform in transforms:
            mat = transform_matrix(transform, imp, data.logcounts, data.counts)
            grid_rows = []
            for cseed in cluster_seeds:
                grid_rows.extend(
                    phase5.cluster_grid(
                        mat,
                        data.labels,
                        data.batch,
                        n_pcs_values=n_pcs_values,
                        n_neighbors_values=n_neighbors_values,
                        resolutions=resolutions,
                        seed=cseed,
                        silhouette_sample_size=5000,
                    )
                )
            summary = summarize_grid(grid_rows, n_labels=n_labels)
            row = {
                "dataset": data.name,
                "config": name,
                "transform": transform,
                "seed": args.seed,
                "cluster_seeds": args.cluster_seeds,
                "runtime_sec": runtime,
                **summary,
            }
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
    out = args.out_dir / f"{data.name}_screen.tsv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, delimiter="\t", fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
