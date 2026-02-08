#!/usr/bin/env python3
from __future__ import annotations

import csv
import os
import subprocess
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clustering_eval import evaluate_clustering
from rds2py import read_rds


SYNTH_INPUT = "synthetic_datasets/rds_splat_output/cells_100"
REAL_INPUT = "datasets"
BENCH_DATASETWISE = "tmp_impute_eval/options_focused5/benchmarks_5methods_datasetwise.tsv"
BENCH_SUMMARY = "tmp_impute_eval/options_focused5/benchmarks_5methods_summary.tsv"
BLENDS = (0.0, 0.2, 0.5)
SEEDS = (42, 43, 44)


@dataclass(frozen=True)
class Config:
    tag: str
    args: Tuple[str, ...]


OPTION_CONFIGS: Dict[int, List[Config]] = {
    1: [
        Config(
            "o1_bio_strict",
            (
                "--loss-bio-weight",
                "3.4",
                "--bio-reg-weight",
                "2.0",
                "--p-nz",
                "0.30",
                "--noise-max",
                "0.12",
                "--prior-blend",
                "0.85",
                "--update-min-dropout",
                "0.95",
                "--gate-loss-weight",
                "0.40",
            ),
        ),
        Config(
            "o1_balanced",
            (
                "--loss-bio-weight",
                "2.8",
                "--bio-reg-weight",
                "1.4",
                "--p-nz",
                "0.38",
                "--noise-max",
                "0.18",
                "--prior-blend",
                "0.75",
                "--update-min-dropout",
                "0.90",
                "--gate-loss-weight",
                "0.50",
            ),
        ),
        Config(
            "o1_struct",
            (
                "--loss-bio-weight",
                "3.2",
                "--bio-reg-weight",
                "1.8",
                "--p-nz",
                "0.32",
                "--noise-max",
                "0.14",
                "--prior-blend",
                "0.83",
                "--update-min-dropout",
                "0.94",
                "--gate-loss-weight",
                "0.45",
                "--gate-margin",
                "0.40",
                "--gate-margin-weight",
                "0.20",
                "--latent-consistency-weight",
                "0.04",
                "--full-recon-weight",
                "0.03",
                "--cell-sim-weight",
                "0.015",
            ),
        ),
    ],
    2: [
        Config(
            "o2_gate_strict",
            (
                "--loss-bio-weight",
                "4.2",
                "--bio-reg-weight",
                "2.4",
                "--p-nz",
                "0.24",
                "--noise-max",
                "0.09",
                "--prior-blend",
                "0.90",
                "--update-min-dropout",
                "0.96",
                "--refine-alpha",
                "0.34",
                "--p-zero",
                "0.02",
                "--shared-gate-loss-weight",
                "0.35",
                "--refine-gate-blend",
                "0.40",
            ),
        ),
        Config(
            "o2_gate_mid",
            (
                "--loss-bio-weight",
                "3.6",
                "--bio-reg-weight",
                "1.9",
                "--p-nz",
                "0.30",
                "--noise-max",
                "0.12",
                "--prior-blend",
                "0.86",
                "--update-min-dropout",
                "0.94",
                "--refine-alpha",
                "0.30",
                "--shared-gate-loss-weight",
                "0.30",
                "--refine-gate-blend",
                "0.35",
            ),
        ),
        Config(
            "o2_gate_struct",
            (
                "--loss-bio-weight",
                "4.4",
                "--bio-reg-weight",
                "2.6",
                "--p-nz",
                "0.20",
                "--noise-max",
                "0.08",
                "--prior-blend",
                "0.92",
                "--update-min-dropout",
                "0.97",
                "--refine-alpha",
                "0.36",
                "--p-zero",
                "0.02",
                "--shared-gate-loss-weight",
                "0.38",
                "--refine-gate-blend",
                "0.45",
                "--latent-consistency-weight",
                "0.04",
                "--full-recon-weight",
                "0.03",
                "--cell-sim-weight",
                "0.015",
            ),
        ),
    ],
    3: [
        Config(
            "o3_balanced",
            (
                "--loss-bio-weight",
                "2.9",
                "--bio-reg-weight",
                "1.4",
                "--p-nz",
                "0.38",
                "--noise-max",
                "0.18",
                "--gate-loss-weight",
                "0.50",
                "--expert-bio-weight",
                "0.55",
                "--expert-drop-weight",
                "0.16",
                "--prior-blend",
                "0.76",
                "--update-min-dropout",
                "0.90",
            ),
        ),
        Config(
            "o3_gate",
            (
                "--loss-bio-weight",
                "2.5",
                "--bio-reg-weight",
                "1.2",
                "--p-nz",
                "0.42",
                "--noise-max",
                "0.21",
                "--gate-loss-weight",
                "0.75",
                "--gate-margin",
                "0.50",
                "--gate-margin-weight",
                "0.25",
                "--expert-bio-weight",
                "0.42",
                "--expert-drop-weight",
                "0.22",
                "--prior-blend",
                "0.70",
                "--update-min-dropout",
                "0.88",
            ),
        ),
        Config(
            "o3_struct",
            (
                "--loss-bio-weight",
                "2.9",
                "--bio-reg-weight",
                "1.4",
                "--p-nz",
                "0.38",
                "--noise-max",
                "0.18",
                "--gate-loss-weight",
                "0.50",
                "--expert-bio-weight",
                "0.55",
                "--expert-drop-weight",
                "0.16",
                "--prior-blend",
                "0.76",
                "--update-min-dropout",
                "0.90",
                "--gate-margin",
                "0.35",
                "--gate-margin-weight",
                "0.15",
                "--latent-consistency-weight",
                "0.05",
                "--full-recon-weight",
                "0.05",
                "--cell-sim-weight",
                "0.020",
            ),
        ),
    ],
    4: [
        Config(
            "o4_gate_bio",
            (
                "--loss-bio-weight",
                "4.1",
                "--bio-reg-weight",
                "2.2",
                "--p-nz",
                "0.26",
                "--noise-max",
                "0.10",
                "--knn-k",
                "6",
                "--prior-blend",
                "0.88",
                "--update-min-dropout",
                "0.96",
                "--refine-alpha",
                "0.34",
                "--shared-gate-loss-weight",
                "0.35",
                "--refine-gate-blend",
                "0.40",
            ),
        ),
        Config(
            "o4_gate_mid",
            (
                "--loss-bio-weight",
                "3.4",
                "--bio-reg-weight",
                "1.8",
                "--p-nz",
                "0.32",
                "--noise-max",
                "0.13",
                "--knn-k",
                "8",
                "--prior-blend",
                "0.82",
                "--update-min-dropout",
                "0.93",
                "--refine-alpha",
                "0.30",
                "--shared-gate-loss-weight",
                "0.30",
                "--refine-gate-blend",
                "0.35",
            ),
        ),
        Config(
            "o4_gate_struct",
            (
                "--loss-bio-weight",
                "4.4",
                "--bio-reg-weight",
                "2.4",
                "--p-nz",
                "0.22",
                "--noise-max",
                "0.09",
                "--knn-k",
                "6",
                "--prior-blend",
                "0.90",
                "--update-min-dropout",
                "0.96",
                "--refine-alpha",
                "0.34",
                "--shared-gate-loss-weight",
                "0.35",
                "--refine-gate-blend",
                "0.40",
                "--latent-consistency-weight",
                "0.04",
                "--full-recon-weight",
                "0.03",
                "--cell-sim-weight",
                "0.015",
            ),
        ),
    ],
    5: [
        Config(
            "o5_bio",
            (
                "--loss-bio-weight",
                "3.9",
                "--bio-reg-weight",
                "2.1",
                "--p-nz",
                "0.28",
                "--noise-max",
                "0.10",
                "--gene-shrink",
                "0.78",
                "--prior-blend",
                "0.88",
                "--update-min-dropout",
                "0.96",
                "--shared-gate-loss-weight",
                "0.30",
                "--refine-gate-blend",
                "0.35",
            ),
        ),
        Config(
            "o5_tight",
            (
                "--loss-bio-weight",
                "4.2",
                "--bio-reg-weight",
                "2.6",
                "--p-nz",
                "0.22",
                "--noise-max",
                "0.08",
                "--gene-shrink",
                "0.80",
                "--prior-blend",
                "0.90",
                "--update-min-dropout",
                "0.97",
                "--shared-gate-loss-weight",
                "0.35",
                "--refine-gate-blend",
                "0.40",
            ),
        ),
        Config(
            "o5_struct",
            (
                "--loss-bio-weight",
                "4.0",
                "--bio-reg-weight",
                "2.2",
                "--p-nz",
                "0.26",
                "--noise-max",
                "0.09",
                "--gene-shrink",
                "0.80",
                "--prior-blend",
                "0.90",
                "--update-min-dropout",
                "0.96",
                "--shared-gate-loss-weight",
                "0.33",
                "--refine-gate-blend",
                "0.38",
                "--latent-consistency-weight",
                "0.04",
                "--full-recon-weight",
                "0.03",
                "--cell-sim-weight",
                "0.015",
            ),
        ),
    ],
}


def write_tsv(path: Path, rows: Iterable[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def read_single_row_tsv(path: Path) -> Dict[str, str]:
    with path.open() as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            return dict(row)
    raise RuntimeError(f"No rows in {path}")


def extract_labels(dataset_name: str) -> np.ndarray:
    sce = read_rds(str(REPO_ROOT / REAL_INPUT / f"{dataset_name}.rds"))
    colmd = getattr(sce, "column_data", None) or getattr(sce, "colData", None)
    y = None
    for key in ("cell_type1", "labels", "Group", "label"):
        try:
            y = np.asarray(colmd.get_column(key)) if hasattr(colmd, "get_column") else np.asarray(colmd[key])
            break
        except Exception:
            continue
    if y is None:
        raise RuntimeError(f"Missing labels for {dataset_name}")
    _, lab = np.unique(y, return_inverse=True)
    return lab.astype(np.int32)


def run_cmd(cmd: List[str], log_path: Path, gpu_id: int) -> None:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env.setdefault("OMP_NUM_THREADS", "8")
    env.setdefault("MKL_NUM_THREADS", "8")
    env.setdefault("OPENBLAS_NUM_THREADS", "8")
    env.setdefault("NUMEXPR_NUM_THREADS", "8")
    env.setdefault("TORCH_NUM_THREADS", "8")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as f:
        f.write("CMD: " + " ".join(cmd) + "\n")
        f.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")


def run_option_worker(option: int, gpu_id: int, root_dir: str) -> Dict[str, object]:
    option_dir = Path(root_dir) / f"option{option}"
    option_dir.mkdir(parents=True, exist_ok=True)
    script = f"masked_imputation_option{option}.py"
    phase_rows: List[Dict[str, object]] = []

    for cfg in OPTION_CONFIGS[option]:
        out = option_dir / cfg.tag
        cmd = [
            "python",
            script,
            SYNTH_INPUT,
            str(out),
            "--device",
            "cuda",
            "--seed",
            "42",
            "--keep-positive",
            "true",
            "--save-imputed",
            "false",
            "--epochs",
            "180",
            "--batch-size",
            "64",
            *cfg.args,
        ]
        run_cmd(cmd, out / "run.log", gpu_id=gpu_id)
        row = read_single_row_tsv(out / f"masked_imputation_option{option}_summary.tsv")
        avg_mse = float(row["avg_mse"])
        avg_bio = float(row["avg_biozero"])
        feasible = (avg_mse <= 1.0) and (avg_bio <= 0.2)
        gap = max(0.0, avg_mse - 1.0) + max(0.0, avg_bio - 0.2)
        phase_rows.append(
            {
                "tag": cfg.tag,
                "avg_mse": avg_mse,
                "avg_biozero": avg_bio,
                "feasible": feasible,
                "gap": gap,
            }
        )

    phase_rows_sorted = sorted(
        phase_rows,
        key=lambda r: (
            0 if bool(r["feasible"]) else 1,
            float(r["gap"]),
            float(r["avg_biozero"]),
            float(r["avg_mse"]),
        ),
    )
    best_phase = phase_rows_sorted[0]
    base_tag = str(best_phase["tag"])
    base_cfg = next(c for c in OPTION_CONFIGS[option] if c.tag == base_tag)
    write_tsv(
        option_dir / "phase_a.tsv",
        phase_rows,
        ["tag", "avg_mse", "avg_biozero", "feasible", "gap"],
    )

    real_out = option_dir / f"{base_tag}_real"
    real_cmd = [
        "python",
        script,
        REAL_INPUT,
        str(real_out),
        "--device",
        "cuda",
        "--seed",
        "42",
        "--keep-positive",
        "false",
        "--save-imputed",
        "true",
        "--epochs",
        "90",
        "--batch-size",
        "128",
        "--real-norm-mode",
        "median",
        "--orig-blend",
        "0.0",
        *base_cfg.args,
    ]
    run_cmd(real_cmd, real_out / "run.log", gpu_id=gpu_id)

    labels_by_ds = {p.stem: extract_labels(p.stem) for p in (REPO_ROOT / REAL_INPUT).glob("*.rds")}
    ari_metrics: List[Dict[str, object]] = []
    for npz_path in sorted(real_out.glob("*_imputed.npz")):
        ds = npz_path.stem.replace("_imputed", "")
        arr = np.load(npz_path)
        imp = arr["log_imputed"].astype(np.float32)
        obs = arr["logcounts"].astype(np.float32)
        y = labels_by_ds[ds]
        for blend in BLENDS:
            x = ((1.0 - blend) * imp + blend * obs).astype(np.float32)
            for seed in SEEDS:
                res = evaluate_clustering(x, y, seed=int(seed))
                ari_metrics.append(
                    {
                        "option": f"option{option}",
                        "base_tag": base_tag,
                        "blend": blend,
                        "dataset": ds,
                        "seed": int(seed),
                        "ARI": float(res["ARI"]),
                        "ASW": float(res["ASW"]),
                        "NMI": float(res["NMI"]),
                        "PS": float(res["PS"]),
                    }
                )

    write_tsv(
        option_dir / "ari_metrics.tsv",
        ari_metrics,
        ["option", "base_tag", "blend", "dataset", "seed", "ARI", "ASW", "NMI", "PS"],
    )

    ari_summary: List[Dict[str, object]] = []
    for blend in BLENDS:
        rows = [r for r in ari_metrics if abs(float(r["blend"]) - float(blend)) < 1e-12]
        ari_vals = [float(r["ARI"]) for r in rows]
        ari_summary.append(
            {
                "option": f"option{option}",
                "base_tag": base_tag,
                "blend": blend,
                "avg_ari": float(np.mean(ari_vals)) if ari_vals else float("nan"),
                "n_scores": len(rows),
                "n_datasets": len({str(r["dataset"]) for r in rows}),
                "n_seeds": len({int(r["seed"]) for r in rows}),
            }
        )
    write_tsv(
        option_dir / "ari_summary.tsv",
        ari_summary,
        ["option", "base_tag", "blend", "avg_ari", "n_scores", "n_datasets", "n_seeds"],
    )

    best_blend_row = max(ari_summary, key=lambda r: float(r["avg_ari"]))
    result_row = {
        "option": f"option{option}",
        "base_tag": base_tag,
        "avg_mse": float(best_phase["avg_mse"]),
        "avg_biozero": float(best_phase["avg_biozero"]),
        "feasible": bool(best_phase["feasible"]),
        "best_blend": float(best_blend_row["blend"]),
        "avg_ari": float(best_blend_row["avg_ari"]),
    }
    write_tsv(
        option_dir / "results.tsv",
        [result_row],
        ["option", "base_tag", "avg_mse", "avg_biozero", "feasible", "best_blend", "avg_ari"],
    )
    return result_row


def load_benchmark_tables() -> Tuple[Dict[str, float], float]:
    by_dataset: Dict[str, float] = {}
    with (REPO_ROOT / BENCH_DATASETWISE).open() as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            ds = str(row["dataset"])
            ari = float(row["ARI"])
            by_dataset[ds] = max(by_dataset.get(ds, -np.inf), ari)

    best_avg = -np.inf
    with (REPO_ROOT / BENCH_SUMMARY).open() as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            best_avg = max(best_avg, float(row["avg_ari"]))
    return by_dataset, float(best_avg)


def summarize_global(root_dir: Path, option_rows: List[Dict[str, object]]) -> None:
    bench_by_ds, best_bench_avg = load_benchmark_tables()
    target_5pct = 1.05 * best_bench_avg

    all_rows: List[Dict[str, object]] = []
    for row in option_rows:
        opt = str(row["option"])
        option_dir = root_dir / opt
        base_tag = str(row["base_tag"])
        best_blend = float(row["best_blend"])
        ari_metrics = []
        with (option_dir / "ari_metrics.tsv").open() as f:
            r = csv.DictReader(f, delimiter="\t")
            for rr in r:
                if abs(float(rr["blend"]) - best_blend) < 1e-12:
                    ari_metrics.append(rr)
        ds_mean: Dict[str, float] = {}
        for ds in sorted({str(r["dataset"]) for r in ari_metrics}):
            vals = [float(r["ARI"]) for r in ari_metrics if str(r["dataset"]) == ds]
            ds_mean[ds] = float(np.mean(vals)) if vals else float("nan")
        beat_count = int(sum(ds_mean.get(ds, -np.inf) > bench_by_ds.get(ds, np.inf) for ds in bench_by_ds))
        feasible = bool(row["feasible"])
        avg_ari = float(row["avg_ari"])
        meets3 = beat_count >= 3
        better5pct = avg_ari >= target_5pct
        objective = feasible and meets3 and better5pct
        all_rows.append(
            {
                "option": opt,
                "base_tag": base_tag,
                "avg_mse": float(row["avg_mse"]),
                "avg_biozero": float(row["avg_biozero"]),
                "feasible": feasible,
                "best_blend": best_blend,
                "avg_ari": avg_ari,
                "beat_count_5": beat_count,
                "meets_3_of_5": meets3,
                "better_5pct_than_best_benchmark": better5pct,
                "overall_objective_met": objective,
            }
        )

    write_tsv(
        root_dir / "focused7_all_rows.tsv",
        all_rows,
        [
            "option",
            "base_tag",
            "avg_mse",
            "avg_biozero",
            "feasible",
            "best_blend",
            "avg_ari",
            "beat_count_5",
            "meets_3_of_5",
            "better_5pct_than_best_benchmark",
            "overall_objective_met",
        ],
    )

    by_option = sorted(all_rows, key=lambda r: int(str(r["option"]).replace("option", "")))
    write_tsv(
        root_dir / "focused7_best_by_option.tsv",
        by_option,
        [
            "option",
            "base_tag",
            "avg_mse",
            "avg_biozero",
            "feasible",
            "best_blend",
            "avg_ari",
            "beat_count_5",
            "meets_3_of_5",
            "better_5pct_than_best_benchmark",
            "overall_objective_met",
        ],
    )

    feasible_rows = [r for r in all_rows if bool(r["feasible"])]
    best_row = max(feasible_rows if feasible_rows else all_rows, key=lambda r: float(r["avg_ari"]))
    n_meet_3 = sum(bool(r["meets_3_of_5"]) for r in all_rows)
    n_meet_full = sum(bool(r["overall_objective_met"]) for r in all_rows)

    write_tsv(
        root_dir / "focused7_objective_summary.tsv",
        [
            {
                "best_benchmark_method": "ccimpute",
                "best_benchmark_avg_ari": best_bench_avg,
                "target_5pct_over_best": target_5pct,
                "n_options": len(all_rows),
                "n_feasible": len(feasible_rows),
                "n_meet_3_of_5": n_meet_3,
                "n_meet_full_objective": n_meet_full,
                "best_option": best_row["option"],
                "best_tag": best_row["base_tag"],
                "best_blend": best_row["best_blend"],
                "best_avg_ari": best_row["avg_ari"],
                "best_avg_mse": best_row["avg_mse"],
                "best_avg_biozero": best_row["avg_biozero"],
                "best_feasible": best_row["feasible"],
                "best_beat_count_5": best_row["beat_count_5"],
            }
        ],
        [
            "best_benchmark_method",
            "best_benchmark_avg_ari",
            "target_5pct_over_best",
            "n_options",
            "n_feasible",
            "n_meet_3_of_5",
            "n_meet_full_objective",
            "best_option",
            "best_tag",
            "best_blend",
            "best_avg_ari",
            "best_avg_mse",
            "best_avg_biozero",
            "best_feasible",
            "best_beat_count_5",
        ],
    )


def main() -> int:
    root_dir = REPO_ROOT / "tmp_impute_eval" / "options_focused7"
    root_dir.mkdir(parents=True, exist_ok=True)
    gpu_by_option = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}

    option_rows: List[Dict[str, object]] = []
    failures: List[str] = []

    with ProcessPoolExecutor(max_workers=5) as ex:
        fut_to_opt = {
            ex.submit(run_option_worker, opt, gpu_by_option[opt], str(root_dir)): opt for opt in sorted(OPTION_CONFIGS)
        }
        for fut in as_completed(fut_to_opt):
            opt = fut_to_opt[fut]
            try:
                row = fut.result()
                option_rows.append(row)
            except Exception:
                failures.append(f"option{opt}")
                (root_dir / f"option{opt}" / "worker_error.log").write_text(traceback.format_exc())

    if failures:
        print("Failed options:", ", ".join(failures))
    if option_rows:
        summarize_global(root_dir, option_rows)
        print("Wrote focused7 outputs under", root_dir)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
