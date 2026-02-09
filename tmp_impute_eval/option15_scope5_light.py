#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from rds2py import read_rds

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clustering_eval import evaluate_clustering

DATASETS_DIR = Path(os.getenv("CLUSTER_DATASETS_DIR", "datasets"))
BENCH_PY_DIR = Path(os.getenv("BENCH_PY_DIR", "results_clustering_py/scope_dynamic_v1"))
BENCH_R_DIR = Path(os.getenv("BENCH_R_DIR", "results_clustering_r/scope_dynamic_v1"))
OUT_ROOT = Path(os.getenv("OPTIONS_OUT_ROOT", "tmp_impute_eval/options_scope_dynamic_v1"))
CONFIG_FILTER = os.getenv("CONFIG_FILTER", "").strip()
EVAL_SEEDS = tuple(
    int(x.strip()) for x in os.getenv("EVAL_SEEDS", "42,43,44").split(",") if x.strip()
)

CONFIGS: List[Dict[str, object]] = [
    {"option": 1, "name": "option1_default", "args": []},
    {
        "option": 1,
        "name": "option1_biofocus",
        "args": [
            "--loss-bio-weight",
            "2.8",
            "--bio-reg-weight",
            "1.4",
            "--gate-loss-weight",
            "0.8",
            "--prior-blend",
            "0.75",
            "--update-min-dropout",
            "0.8",
            "--refine-alpha",
            "0.3",
            "--p-nz",
            "0.4",
        ],
    },
    {"option": 2, "name": "option2_default", "args": []},
    {
        "option": 2,
        "name": "option2_biofocus",
        "args": [
            "--loss-bio-weight",
            "3.0",
            "--bio-reg-weight",
            "1.5",
            "--prior-blend",
            "0.8",
            "--update-min-dropout",
            "0.8",
            "--refine-alpha",
            "0.3",
            "--p-nz",
            "0.35",
        ],
    },
    {"option": 3, "name": "option3_default", "args": []},
    {
        "option": 3,
        "name": "option3_biofocus",
        "args": [
            "--loss-bio-weight",
            "2.8",
            "--bio-reg-weight",
            "1.4",
            "--gate-loss-weight",
            "0.65",
            "--expert-bio-weight",
            "0.55",
            "--expert-drop-weight",
            "0.15",
            "--prior-blend",
            "0.75",
            "--update-min-dropout",
            "0.8",
            "--p-nz",
            "0.4",
        ],
    },
    {
        "option": 3,
        "name": "option3_midbio",
        "args": [
            "--loss-bio-weight",
            "2.35",
            "--bio-reg-weight",
            "1.1",
            "--gate-loss-weight",
            "0.55",
            "--expert-bio-weight",
            "0.45",
            "--expert-drop-weight",
            "0.2",
            "--prior-blend",
            "0.72",
            "--update-min-dropout",
            "0.83",
            "--p-nz",
            "0.42",
        ],
    },
    {
        "option": 3,
        "name": "option3_ari",
        "args": [
            "--loss-bio-weight",
            "2.1",
            "--bio-reg-weight",
            "1.0",
            "--gate-loss-weight",
            "0.5",
            "--expert-bio-weight",
            "0.35",
            "--expert-drop-weight",
            "0.2",
            "--prior-blend",
            "0.68",
            "--update-min-dropout",
            "0.85",
            "--p-nz",
            "0.45",
        ],
    },
    {
        "option": 3,
        "name": "option3_labelcal_beta03",
        "args": [
            "--loss-bio-weight",
            "2.8",
            "--bio-reg-weight",
            "1.4",
            "--gate-loss-weight",
            "0.65",
            "--expert-bio-weight",
            "0.55",
            "--expert-drop-weight",
            "0.15",
            "--prior-blend",
            "0.75",
            "--update-min-dropout",
            "0.8",
            "--p-nz",
            "0.4",
            "--label-calibrate-beta",
            "0.3",
        ],
    },
    {
        "option": 3,
        "name": "option3_labelcal_beta04",
        "args": [
            "--loss-bio-weight",
            "2.8",
            "--bio-reg-weight",
            "1.4",
            "--gate-loss-weight",
            "0.65",
            "--expert-bio-weight",
            "0.55",
            "--expert-drop-weight",
            "0.15",
            "--prior-blend",
            "0.75",
            "--update-min-dropout",
            "0.8",
            "--p-nz",
            "0.4",
            "--label-calibrate-beta",
            "0.4",
        ],
    },
    {
        "option": 3,
        "name": "option3_labelcal_auto",
        "args": [
            "--loss-bio-weight",
            "2.8",
            "--bio-reg-weight",
            "1.4",
            "--gate-loss-weight",
            "0.65",
            "--expert-bio-weight",
            "0.55",
            "--expert-drop-weight",
            "0.15",
            "--prior-blend",
            "0.75",
            "--update-min-dropout",
            "0.8",
            "--p-nz",
            "0.4",
            "--label-calibrate-beta",
            "auto",
            "--label-auto-clusterer",
            "simple",
        ],
    },
    {
        "option": 3,
        "name": "option3_labelcal_auto_struct1",
        "args": [
            "--loss-bio-weight",
            "2.8",
            "--bio-reg-weight",
            "1.4",
            "--gate-loss-weight",
            "0.65",
            "--expert-bio-weight",
            "0.55",
            "--expert-drop-weight",
            "0.15",
            "--prior-blend",
            "0.75",
            "--update-min-dropout",
            "0.8",
            "--p-nz",
            "0.4",
            "--label-calibrate-beta",
            "auto",
            "--label-auto-clusterer",
            "simple",
            "--label-auto-k-min",
            "3",
            "--label-auto-k-max",
            "16",
            "--label-auto-beta-min",
            "0.55",
            "--label-auto-beta-max",
            "1.0",
            "--label-auto-conf-floor",
            "0.25",
            "--label-auto-n0",
            "8",
            "--label-auto-purity-center",
            "0.5",
            "--gate-margin",
            "0.35",
            "--gate-margin-weight",
            "0.15",
            "--latent-consistency-weight",
            "0.04",
            "--full-recon-weight",
            "0.04",
            "--cell-sim-weight",
            "0.015",
        ],
    },
    {
        "option": 3,
        "name": "option3_labelcal_auto_struct2",
        "args": [
            "--loss-bio-weight",
            "2.9",
            "--bio-reg-weight",
            "1.45",
            "--gate-loss-weight",
            "0.6",
            "--expert-bio-weight",
            "0.6",
            "--expert-drop-weight",
            "0.14",
            "--prior-blend",
            "0.78",
            "--update-min-dropout",
            "0.82",
            "--p-nz",
            "0.38",
            "--label-calibrate-beta",
            "auto",
            "--label-auto-clusterer",
            "simple",
            "--label-auto-k-min",
            "3",
            "--label-auto-k-max",
            "18",
            "--label-auto-beta-min",
            "0.6",
            "--label-auto-beta-max",
            "1.0",
            "--label-auto-conf-floor",
            "0.3",
            "--label-auto-n0",
            "6",
            "--label-auto-purity-center",
            "0.45",
            "--gate-margin",
            "0.4",
            "--gate-margin-weight",
            "0.18",
            "--latent-consistency-weight",
            "0.05",
            "--full-recon-weight",
            "0.05",
            "--cell-sim-weight",
            "0.02",
        ],
    },
    {
        "option": 3,
        "name": "option3_labelcal_auto_structlight",
        "args": [
            "--loss-bio-weight",
            "2.8",
            "--bio-reg-weight",
            "1.4",
            "--gate-loss-weight",
            "0.65",
            "--expert-bio-weight",
            "0.55",
            "--expert-drop-weight",
            "0.15",
            "--prior-blend",
            "0.75",
            "--update-min-dropout",
            "0.8",
            "--p-nz",
            "0.4",
            "--label-calibrate-beta",
            "auto",
            "--label-auto-clusterer",
            "simple",
            "--gate-margin",
            "0.2",
            "--gate-margin-weight",
            "0.08",
            "--latent-consistency-weight",
            "0.02",
            "--full-recon-weight",
            "0.02",
            "--cell-sim-weight",
            "0.005",
        ],
    },
    {
        "option": 3,
        "name": "option3_labelcal_auto_structlight_centerrow",
        "args": [
            "--loss-bio-weight",
            "2.8",
            "--bio-reg-weight",
            "1.4",
            "--gate-loss-weight",
            "0.65",
            "--expert-bio-weight",
            "0.55",
            "--expert-drop-weight",
            "0.15",
            "--prior-blend",
            "0.75",
            "--update-min-dropout",
            "0.8",
            "--p-nz",
            "0.4",
            "--label-calibrate-beta",
            "auto",
            "--label-auto-clusterer",
            "simple",
            "--gate-margin",
            "0.2",
            "--gate-margin-weight",
            "0.08",
            "--latent-consistency-weight",
            "0.02",
            "--full-recon-weight",
            "0.02",
            "--cell-sim-weight",
            "0.005",
            "--real-output-transform",
            "center_row",
        ],
    },
    {
        "option": 3,
        "name": "option3_labelcal_auto_structlight_centerrow_exactk",
        "args": [
            "--loss-bio-weight",
            "2.8",
            "--bio-reg-weight",
            "1.4",
            "--gate-loss-weight",
            "0.65",
            "--expert-bio-weight",
            "0.55",
            "--expert-drop-weight",
            "0.15",
            "--prior-blend",
            "0.75",
            "--update-min-dropout",
            "0.8",
            "--p-nz",
            "0.4",
            "--label-calibrate-beta",
            "auto",
            "--label-auto-clusterer",
            "simple",
            "--label-auto-use-exact-k",
            "true",
            "--gate-margin",
            "0.2",
            "--gate-margin-weight",
            "0.08",
            "--latent-consistency-weight",
            "0.02",
            "--full-recon-weight",
            "0.02",
            "--cell-sim-weight",
            "0.005",
            "--real-output-transform",
            "center_row",
        ],
    },
    {
        "option": 3,
        "name": "option3_labelcal_auto_structlight_centerrow_pearson",
        "args": [
            "--loss-bio-weight",
            "2.8",
            "--bio-reg-weight",
            "1.4",
            "--gate-loss-weight",
            "0.65",
            "--expert-bio-weight",
            "0.55",
            "--expert-drop-weight",
            "0.15",
            "--prior-blend",
            "0.75",
            "--update-min-dropout",
            "0.8",
            "--p-nz",
            "0.4",
            "--label-calibrate-beta",
            "auto",
            "--label-auto-clusterer",
            "simple",
            "--label-auto-metric",
            "pearson",
            "--gate-margin",
            "0.2",
            "--gate-margin-weight",
            "0.08",
            "--latent-consistency-weight",
            "0.02",
            "--full-recon-weight",
            "0.02",
            "--cell-sim-weight",
            "0.005",
            "--real-output-transform",
            "center_row",
        ],
    },
    {
        "option": 3,
        "name": "option3_labelcal_auto_structlight_centerrow_hw1000",
        "args": [
            "--loss-bio-weight",
            "2.8",
            "--bio-reg-weight",
            "1.4",
            "--gate-loss-weight",
            "0.65",
            "--expert-bio-weight",
            "0.55",
            "--expert-drop-weight",
            "0.15",
            "--prior-blend",
            "0.75",
            "--update-min-dropout",
            "0.8",
            "--p-nz",
            "0.4",
            "--label-calibrate-beta",
            "auto",
            "--label-auto-clusterer",
            "hartigan_wong",
            "--label-auto-hw-nstart",
            "1000",
            "--label-auto-hw-iter-max",
            "1000",
            "--gate-margin",
            "0.2",
            "--gate-margin-weight",
            "0.08",
            "--latent-consistency-weight",
            "0.02",
            "--full-recon-weight",
            "0.02",
            "--cell-sim-weight",
            "0.005",
            "--real-output-transform",
            "center_row",
        ],
    },
    {
        "option": 3,
        "name": "option3_labelcal_auto_structlight_centerrow_gmm",
        "args": [
            "--loss-bio-weight",
            "2.8",
            "--bio-reg-weight",
            "1.4",
            "--gate-loss-weight",
            "0.65",
            "--expert-bio-weight",
            "0.55",
            "--expert-drop-weight",
            "0.15",
            "--prior-blend",
            "0.75",
            "--update-min-dropout",
            "0.8",
            "--p-nz",
            "0.4",
            "--label-calibrate-beta",
            "auto",
            "--label-auto-clusterer",
            "gmm",
            "--label-auto-gmm-n-init",
            "64",
            "--label-auto-gmm-max-iter",
            "500",
            "--label-auto-gmm-covariance",
            "full",
            "--label-auto-gmm-reg-covar",
            "1e-5",
            "--gate-margin",
            "0.2",
            "--gate-margin-weight",
            "0.08",
            "--latent-consistency-weight",
            "0.02",
            "--full-recon-weight",
            "0.02",
            "--cell-sim-weight",
            "0.005",
            "--real-output-transform",
            "center_row",
        ],
    },
    {
        "option": 3,
        "name": "option3_labelcal_auto_structlight_centerrow_2_keep",
        "args": [
            "--loss-bio-weight",
            "2.8",
            "--bio-reg-weight",
            "1.4",
            "--gate-loss-weight",
            "0.65",
            "--expert-bio-weight",
            "0.55",
            "--expert-drop-weight",
            "0.15",
            "--prior-blend",
            "0.75",
            "--update-min-dropout",
            "0.8",
            "--p-nz",
            "0.4",
            "--label-calibrate-beta",
            "auto",
            "--label-auto-clusterer",
            "simple",
            "--label-auto-use-exact-k",
            "true",
            "--label-auto-metric",
            "pearson",
            "--gate-margin",
            "0.2",
            "--gate-margin-weight",
            "0.08",
            "--latent-consistency-weight",
            "0.02",
            "--full-recon-weight",
            "0.02",
            "--cell-sim-weight",
            "0.005",
            "--real-output-transform",
            "center_row",
            "--keep-positive",
            "true",
        ],
    },
    {
        "option": 3,
        "name": "option3_labelcal_auto_structlight_centerrow_2",
        "args": [
            "--loss-bio-weight",
            "2.8",
            "--bio-reg-weight",
            "1.4",
            "--gate-loss-weight",
            "0.65",
            "--expert-bio-weight",
            "0.55",
            "--expert-drop-weight",
            "0.15",
            "--prior-blend",
            "0.75",
            "--update-min-dropout",
            "0.8",
            "--p-nz",
            "0.4",
            "--label-calibrate-beta",
            "auto",
            "--label-auto-clusterer",
            "simple",
            "--label-auto-use-exact-k",
            "true",
            "--label-auto-metric",
            "pearson",
            "--gate-margin",
            "0.2",
            "--gate-margin-weight",
            "0.08",
            "--latent-consistency-weight",
            "0.02",
            "--full-recon-weight",
            "0.02",
            "--cell-sim-weight",
            "0.005",
            "--real-output-transform",
            "center_row",
            "--keep-positive",
            "false",
        ],
    },
    {
        "option": 3,
        "name": "option3_labelcal_auto_structlight_centerrow_2_tuneA",
        "args": [
            "--loss-bio-weight",
            "2.8",
            "--bio-reg-weight",
            "1.4",
            "--gate-loss-weight",
            "0.65",
            "--expert-bio-weight",
            "0.55",
            "--expert-drop-weight",
            "0.15",
            "--prior-blend",
            "0.75",
            "--update-min-dropout",
            "0.8",
            "--p-nz",
            "0.4",
            "--label-calibrate-beta",
            "auto",
            "--label-auto-clusterer",
            "simple",
            "--label-auto-use-exact-k",
            "true",
            "--label-auto-metric",
            "pearson",
            "--label-auto-conf-floor",
            "0.10",
            "--gate-margin",
            "0.15",
            "--gate-margin-weight",
            "0.06",
            "--latent-consistency-weight",
            "0.02",
            "--full-recon-weight",
            "0.02",
            "--cell-sim-weight",
            "0.005",
            "--real-output-transform",
            "center_row",
            "--keep-positive",
            "false",
        ],
    },
    {
        "option": 3,
        "name": "option3_labelcal_auto_structlight_centerrow_2_tuneB",
        "args": [
            "--loss-bio-weight",
            "2.8",
            "--bio-reg-weight",
            "1.4",
            "--gate-loss-weight",
            "0.65",
            "--expert-bio-weight",
            "0.55",
            "--expert-drop-weight",
            "0.15",
            "--prior-blend",
            "0.75",
            "--update-min-dropout",
            "0.8",
            "--p-nz",
            "0.4",
            "--label-calibrate-beta",
            "auto",
            "--label-auto-clusterer",
            "simple",
            "--label-auto-use-exact-k",
            "true",
            "--label-auto-metric",
            "pearson",
            "--label-auto-beta-min",
            "0.03",
            "--label-auto-beta-max",
            "0.85",
            "--label-auto-n0",
            "30",
            "--gate-margin",
            "0.2",
            "--gate-margin-weight",
            "0.08",
            "--latent-consistency-weight",
            "0.02",
            "--full-recon-weight",
            "0.02",
            "--cell-sim-weight",
            "0.005",
            "--real-output-transform",
            "center_row",
            "--keep-positive",
            "false",
        ],
    },
    {
        "option": 3,
        "name": "option3_labelcal_auto_structlight_rowl2",
        "args": [
            "--loss-bio-weight",
            "2.8",
            "--bio-reg-weight",
            "1.4",
            "--gate-loss-weight",
            "0.65",
            "--expert-bio-weight",
            "0.55",
            "--expert-drop-weight",
            "0.15",
            "--prior-blend",
            "0.75",
            "--update-min-dropout",
            "0.8",
            "--p-nz",
            "0.4",
            "--label-calibrate-beta",
            "auto",
            "--label-auto-clusterer",
            "simple",
            "--gate-margin",
            "0.2",
            "--gate-margin-weight",
            "0.08",
            "--latent-consistency-weight",
            "0.02",
            "--full-recon-weight",
            "0.02",
            "--cell-sim-weight",
            "0.005",
            "--real-output-transform",
            "row_l2",
        ],
    },
    {
        "option": 3,
        "name": "option3_labelcal_auto_graphcluster",
        "args": [
            "--loss-bio-weight",
            "2.8",
            "--bio-reg-weight",
            "1.4",
            "--gate-loss-weight",
            "0.65",
            "--expert-bio-weight",
            "0.55",
            "--expert-drop-weight",
            "0.15",
            "--prior-blend",
            "0.75",
            "--update-min-dropout",
            "0.8",
            "--p-nz",
            "0.4",
            "--label-calibrate-beta",
            "auto",
            "--label-auto-clusterer",
            "simple",
            "--graph-blend",
            "0.03",
            "--graph-k",
            "12",
            "--cluster-blend",
            "0.03",
            "--cluster-k-min",
            "2",
            "--cluster-k-max",
            "14",
        ],
    },
    {"option": 4, "name": "option4_default", "args": []},
    {
        "option": 4,
        "name": "option4_biofocus",
        "args": [
            "--loss-bio-weight",
            "2.4",
            "--bio-reg-weight",
            "1.2",
            "--prior-blend",
            "0.72",
            "--update-min-dropout",
            "0.8",
            "--knn-k",
            "10",
        ],
    },
    {
        "option": 4,
        "name": "option4_midbio",
        "args": [
            "--loss-bio-weight",
            "2.6",
            "--bio-reg-weight",
            "1.3",
            "--prior-blend",
            "0.76",
            "--update-min-dropout",
            "0.8",
            "--knn-k",
            "10",
            "--p-nz",
            "0.45",
        ],
    },
    {
        "option": 4,
        "name": "option4_strongbio",
        "args": [
            "--loss-bio-weight",
            "2.9",
            "--bio-reg-weight",
            "1.5",
            "--prior-blend",
            "0.8",
            "--update-min-dropout",
            "0.78",
            "--refine-alpha",
            "0.35",
            "--knn-k",
            "10",
            "--p-nz",
            "0.42",
        ],
    },
    {"option": 5, "name": "option5_default", "args": []},
    {
        "option": 5,
        "name": "option5_biofocus",
        "args": [
            "--loss-bio-weight",
            "2.6",
            "--bio-reg-weight",
            "1.3",
            "--prior-blend",
            "0.72",
            "--update-min-dropout",
            "0.8",
            "--gene-shrink",
            "0.25",
        ],
    },
]


def _active_configs() -> List[Dict[str, object]]:
    if not CONFIG_FILTER:
        return CONFIGS
    pat = re.compile(CONFIG_FILTER)
    return [cfg for cfg in CONFIGS if pat.search(str(cfg.get("name", "")))]


def _safe_float(v) -> float:
    try:
        x = float(v)
        return x if math.isfinite(x) else float("nan")
    except Exception:
        return float("nan")


def _read_tsv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _dataset_names() -> List[str]:
    names = sorted(p.stem for p in DATASETS_DIR.glob("*.rds"))
    if not names:
        raise SystemExit(f"No .rds files found in {DATASETS_DIR}")
    return names


def _extract_labels(ds_path: Path) -> np.ndarray:
    sce = read_rds(str(ds_path))
    colmd = getattr(sce, "column_data", None) or getattr(sce, "colData", None)
    if colmd is None:
        raise RuntimeError(f"No column_data found in {ds_path.name}")

    for key in ("cell_type1", "labels", "Group", "label"):
        y = None
        try:
            if hasattr(colmd, "column"):
                y = np.asarray(colmd.column(key))
            elif isinstance(colmd, dict) and key in colmd:
                y = np.asarray(colmd[key])
            elif hasattr(colmd, "columns") and key in list(map(str, colmd.columns)):
                y = np.asarray(colmd[key])
        except Exception:
            y = None
        if y is not None:
            _, ids = np.unique(np.asarray(y), return_inverse=True)
            return ids.astype(int)

    raise RuntimeError(f"No supported label key found in {ds_path.name}")


def _load_benchmark_ari(dataset_names: List[str]) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float], Dict[str, float]]:
    by_method: Dict[str, Dict[str, float]] = {}
    for folder in (BENCH_PY_DIR, BENCH_R_DIR):
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*_clustering_table.tsv")):
            fallback_method = path.name.replace("_clustering_table.tsv", "")
            for row in _read_tsv(path):
                method = (row.get("method") or fallback_method).strip().lower()
                if method == "balanced_mse":
                    continue
                ds = row.get("dataset", "").strip()
                if ds not in dataset_names:
                    continue
                ari = _safe_float(row.get("ARI"))
                if not math.isfinite(ari):
                    continue
                by_method.setdefault(method, {})[ds] = ari

    best_by_ds: Dict[str, float] = {}
    for ds in dataset_names:
        vals = [mvals[ds] for mvals in by_method.values() if ds in mvals]
        best_by_ds[ds] = max(vals) if vals else float("nan")

    avg_by_method: Dict[str, float] = {}
    for method, mvals in by_method.items():
        vals = [mvals.get(ds, float("nan")) for ds in dataset_names]
        vals = [v for v in vals if math.isfinite(v)]
        avg_by_method[method] = float(np.mean(vals)) if vals else float("nan")
    return by_method, best_by_ds, avg_by_method


def _run_config(
    cfg: Dict[str, object],
    labels_cache: Dict[str, np.ndarray],
    dataset_names: List[str],
) -> Dict[str, object]:
    option = int(cfg["option"])
    run_name = str(cfg["name"])
    extra_args = [str(x) for x in cfg.get("args", [])]
    out_dir = OUT_ROOT / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"masked_imputation_option{option}_summary.tsv"
    has_all_npz = all((out_dir / f"{ds}_imputed.npz").exists() for ds in dataset_names)
    should_run = not summary_path.exists() or not has_all_npz

    cmd = [
        "python",
        f"masked_imputation_option{option}.py",
        str(DATASETS_DIR),
        str(out_dir),
        "--device",
        "cuda",
        "--seed",
        "42",
        "--epochs",
        "120",
        "--save-imputed",
        "true",
    ] + extra_args

    if should_run:
        print(f"\n[RUN] {run_name}: {' '.join(cmd)}", flush=True)
        proc = subprocess.run(cmd, capture_output=True, text=True)
        (out_dir / "run.log").write_text(
            proc.stdout + "\n\n===== STDERR =====\n\n" + proc.stderr, encoding="utf-8"
        )
        returncode = int(proc.returncode)
    else:
        print(f"\n[SKIP] {run_name}: existing outputs found", flush=True)
        returncode = 0

    summary_rows = _read_tsv(summary_path)
    avg_mse = float("nan")
    avg_biozero = float("nan")
    runtime_sec = float("nan")
    if summary_rows:
        s0 = summary_rows[0]
        avg_mse = _safe_float(s0.get("avg_mse"))
        avg_biozero = _safe_float(s0.get("avg_biozero"))
        runtime_sec = _safe_float(s0.get("runtime_sec"))

    dataset_ari: Dict[str, float] = {}
    for ds in dataset_names:
        npz_path = out_dir / f"{ds}_imputed.npz"
        if not npz_path.exists():
            dataset_ari[ds] = float("nan")
            continue
        try:
            arr = np.load(npz_path)
            log_imputed = np.asarray(arr["log_imputed"], dtype=np.float32)
            labels = labels_cache[ds]
            ari_vals = []
            for seed in EVAL_SEEDS:
                res = evaluate_clustering(log_imputed, labels, seed=seed)
                ari_vals.append(_safe_float(res.get("ARI")))
            dataset_ari[ds] = float(np.nanmean(np.asarray(ari_vals, dtype=float)))
        except Exception:
            dataset_ari[ds] = float("nan")

    ari_vals = [v for v in dataset_ari.values() if math.isfinite(v)]
    avg_ari = float(np.mean(ari_vals)) if ari_vals else float("nan")
    return {
        "run_name": run_name,
        "option": option,
        "returncode": returncode,
        "avg_mse": avg_mse,
        "avg_biozero": avg_biozero,
        "avg_ari": avg_ari,
        "runtime_sec": runtime_sec,
        "dataset_ari": dataset_ari,
        "error": "" if returncode == 0 else f"returncode={returncode}",
    }


def _write_rows(path: Path, header: List[str], rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in header})


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    dataset_names = _dataset_names()

    labels_cache = {
        ds: _extract_labels(DATASETS_DIR / f"{ds}.rds")
        for ds in dataset_names
    }
    benchmark_by_method, benchmark_best_by_ds, benchmark_avg_by_method = _load_benchmark_ari(dataset_names)

    run_results: List[Dict[str, object]] = []
    dataset_rows: List[Dict[str, object]] = []
    configs = _active_configs()
    if not configs:
        raise SystemExit(f"No configs matched CONFIG_FILTER='{CONFIG_FILTER}'")
    print(f"Active configs: {len(configs)} (filter='{CONFIG_FILTER or 'ALL'}')")
    print(f"Eval seeds: {EVAL_SEEDS}")

    for cfg in configs:
        result = _run_config(cfg, labels_cache=labels_cache, dataset_names=dataset_names)
        run_results.append(result)
        for ds in dataset_names:
            ari = _safe_float(result["dataset_ari"].get(ds))
            bench_best = _safe_float(benchmark_best_by_ds.get(ds))
            dataset_rows.append(
                {
                    "run_name": result["run_name"],
                    "option": result["option"],
                    "dataset": ds,
                    "ari": ari,
                    "benchmark_best_ari": bench_best,
                    "beats_or_ties_benchmark_best": int(
                        math.isfinite(ari) and math.isfinite(bench_best) and ari >= bench_best
                    ),
                }
            )

    for row in run_results:
        wins = 0
        for ds in dataset_names:
            ari = _safe_float(row["dataset_ari"].get(ds))
            bench_best = _safe_float(benchmark_best_by_ds.get(ds))
            if math.isfinite(ari) and math.isfinite(bench_best) and ari >= bench_best:
                wins += 1
        row["wins_vs_benchmark_best"] = wins
        row["meets_mse"] = int(math.isfinite(row["avg_mse"]) and row["avg_mse"] <= 1.0)
        row["meets_biozero"] = int(math.isfinite(row["avg_biozero"]) and row["avg_biozero"] <= 0.2)
        row["meets_joint"] = int(row["meets_mse"] and row["meets_biozero"])

    best_by_option: List[Dict[str, object]] = []
    for option in sorted({int(r["option"]) for r in run_results}):
        subset = [r for r in run_results if int(r["option"]) == option]
        subset.sort(
            key=lambda r: (
                int(r.get("meets_joint", 0)),
                _safe_float(r.get("avg_ari")),
                -_safe_float(r.get("avg_biozero")),
                -_safe_float(r.get("avg_mse")),
            ),
            reverse=True,
        )
        best = subset[0]
        best_by_option.append(
            {
                "option": option,
                "best_run_name": best["run_name"],
                "avg_mse": best["avg_mse"],
                "avg_biozero": best["avg_biozero"],
                "avg_ari": best["avg_ari"],
                "wins_vs_benchmark_best": best["wins_vs_benchmark_best"],
                "meets_joint": best["meets_joint"],
            }
        )

    _write_rows(
        OUT_ROOT / "sweep_results.tsv",
        [
            "run_name",
            "option",
            "avg_mse",
            "avg_biozero",
            "avg_ari",
            "wins_vs_benchmark_best",
            "meets_mse",
            "meets_biozero",
            "meets_joint",
            "runtime_sec",
            "returncode",
            "error",
        ],
        run_results,
    )
    _write_rows(
        OUT_ROOT / "sweep_dataset_ari.tsv",
        ["run_name", "option", "dataset", "ari", "benchmark_best_ari", "beats_or_ties_benchmark_best"],
        dataset_rows,
    )
    _write_rows(
        OUT_ROOT / "best_by_option.tsv",
        ["option", "best_run_name", "avg_mse", "avg_biozero", "avg_ari", "wins_vs_benchmark_best", "meets_joint"],
        best_by_option,
    )

    bench_rows = [
        {"method": method, "avg_ari": avg}
        for method, avg in sorted(benchmark_avg_by_method.items(), key=lambda kv: kv[0])
    ]
    _write_rows(OUT_ROOT / "benchmark_avg_ari.tsv", ["method", "avg_ari"], bench_rows)

    print("\n[DONE] Wrote:")
    print(" -", OUT_ROOT / "sweep_results.tsv")
    print(" -", OUT_ROOT / "sweep_dataset_ari.tsv")
    print(" -", OUT_ROOT / "best_by_option.tsv")
    print(" -", OUT_ROOT / "benchmark_avg_ari.tsv")
    print("\nBenchmark methods loaded:", ", ".join(sorted(benchmark_by_method.keys())))


if __name__ == "__main__":
    main()
