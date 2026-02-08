#!/usr/bin/env python3
"""
maskclass_ablation.py

Ablation runner for the shared maskclass balanced_mse pipeline.

This script reuses the existing benchmarking entry points:
- run_clustering.py (ARI)
- run_imputation.py (MSE, mse_biozero)

It performs one-factor-at-a-time ablations and writes publication-ready artifacts:
- ablation_summary.tsv
- ablation_ari_by_dataset.tsv
- ablation_mse_by_dataset.tsv
- ablation_report.md
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from queue import Queue
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class AblationSpec:
    name: str
    description: str
    patch: Dict[str, Dict[str, Any]]


def _parse_float(value: str) -> float:
    v = (value or "").strip()
    if not v:
        return float("nan")
    try:
        return float(v)
    except Exception:
        return float("nan")


def _fmt(x: float, ndigits: int = 4) -> str:
    if not math.isfinite(x):
        return "nan"
    return f"{x:.{ndigits}f}"


def _mean(values: Iterable[float]) -> float:
    xs = [v for v in values if math.isfinite(v)]
    if not xs:
        return float("nan")
    return float(sum(xs) / len(xs))


def _read_tsv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _write_tsv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def _load_maskclass_defaults() -> Dict[str, Dict[str, Any]]:
    old = os.environ.pop("MASKCLASS_CONFIG_JSON", None)
    try:
        mod = importlib.import_module("maskclass")
    finally:
        if old is not None:
            os.environ["MASKCLASS_CONFIG_JSON"] = old

    return {
        "BIO_PARAMS": deepcopy(mod.BIO_PARAMS),
        "MODEL_PARAMS": deepcopy(mod.MODEL_PARAMS),
        "AE_PARAMS": deepcopy(mod.AE_PARAMS),
        "SCALER_PARAMS": deepcopy(mod.SCALER_PARAMS),
        "POSTPROCESS_PARAMS": deepcopy(mod.POSTPROCESS_PARAMS),
    }


def _default_ablations() -> List[AblationSpec]:
    return [
        AblationSpec(
            name="no_non_umi_blend",
            description="Remove empirical non-UMI blending from biological-zero estimation.",
            patch={"BIO_PARAMS": {"non_umi_blend": 0.0}},
        ),
        AblationSpec(
            name="no_non_umi_depth",
            description="Remove cell-depth correction from biological-zero estimation.",
            patch={"BIO_PARAMS": {"non_umi_depth_weight": 0.0}},
        ),
        AblationSpec(
            name="old_cell_zero_weight",
            description="Revert cell-zero weighting to masked_imputation26 setting.",
            patch={"BIO_PARAMS": {"cell_zero_weight": 0.30}},
        ),
        AblationSpec(
            name="old_p_nz",
            description="Revert nonzero masking probability to masked_imputation26 setting.",
            patch={"AE_PARAMS": {"p_nz": 0.30}},
        ),
        AblationSpec(
            name="old_noise_max",
            description="Revert biological-noise amplitude to masked_imputation26 setting.",
            patch={"AE_PARAMS": {"noise_max": 0.20}},
        ),
        AblationSpec(
            name="no_graph_refine",
            description="Disable graph smoothing stage.",
            patch={"POSTPROCESS_PARAMS": {"graph_blend": 0.0}},
        ),
        AblationSpec(
            name="no_diffusion_refine",
            description="Disable diffusion refinement stage.",
            patch={"POSTPROCESS_PARAMS": {"diffusion_blend": 0.0}},
        ),
        AblationSpec(
            name="no_cluster_refine",
            description="Disable cluster-centroid refinement stage.",
            patch={"POSTPROCESS_PARAMS": {"cluster_blend": 0.0}},
        ),
        AblationSpec(
            name="no_biozero_shrink",
            description="Disable shrinkage on biologically-likely zeros.",
            patch={"POSTPROCESS_PARAMS": {"biozero_shrink": 0.0}},
        ),
        AblationSpec(
            name="no_postprocess",
            description="Disable all postprocessing stages and biozero shrinkage.",
            patch={
                "POSTPROCESS_PARAMS": {
                    "graph_blend": 0.0,
                    "diffusion_blend": 0.0,
                    "cluster_blend": 0.0,
                    "biozero_shrink": 0.0,
                }
            },
        ),
    ]


def _merge_config(
    base: Dict[str, Dict[str, Any]],
    patch: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    out = deepcopy(base)
    for grp, updates in patch.items():
        if grp not in out:
            out[grp] = {}
        out[grp].update(updates)
    return out


def _run_cmd(cmd: List[str], cwd: Path, env: Dict[str, str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n\n")
        log.flush()
        p = subprocess.run(cmd, cwd=str(cwd), env=env, stdout=log, stderr=subprocess.STDOUT)
        if p.returncode != 0:
            raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}. See {log_path}")


def _execute_job(
    name: str,
    description: str,
    patch: Optional[Dict[str, Dict[str, Any]]],
    args: argparse.Namespace,
    gpu_id: str,
    root: Path,
) -> Dict[str, Any]:
    run_root = root / "runs" / name
    clustering_out = run_root / "clustering"
    imputation_out = run_root / "imputation"

    clustering_table = clustering_out / "balanced_mse_clustering_table.tsv"
    imputation_table = imputation_out / "balanced_mse_mse_table.tsv"

    if args.reuse_existing and clustering_table.exists() and imputation_table.exists():
        return {
            "name": name,
            "description": description,
            "patch": patch or {},
            "gpu": gpu_id,
            "clustering_table": clustering_table,
            "imputation_table": imputation_table,
        }

    env = os.environ.copy()
    if bool(args.force_cpu):
        env["CUDA_VISIBLE_DEVICES"] = ""
    else:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    thread_str = str(int(args.cpu_threads))
    env["OMP_NUM_THREADS"] = thread_str
    env["MKL_NUM_THREADS"] = thread_str
    env["OPENBLAS_NUM_THREADS"] = thread_str
    env["NUMEXPR_NUM_THREADS"] = thread_str
    env["TORCH_NUM_THREADS"] = thread_str
    env.pop("MASKCLASS_CONFIG_JSON", None)

    if patch:
        env["MASKCLASS_CONFIG_JSON"] = json.dumps(patch, separators=(",", ":"))

    need_clustering = not (args.reuse_existing and clustering_table.exists())
    need_imputation = not (args.reuse_existing and imputation_table.exists())

    if need_clustering:
        _run_cmd(
            [
                args.python,
                "run_clustering.py",
                args.clustering_input,
                str(clustering_out),
                "balanced_mse",
                "--n-repeat",
                str(args.clustering_repeats),
            ],
            cwd=args.repo_root,
            env=env,
            log_path=run_root / "clustering.log",
        )

    if need_imputation:
        _run_cmd(
            [
                args.python,
                "run_imputation.py",
                args.imputation_input,
                str(imputation_out),
                "balanced_mse",
                "--n-repeat",
                str(args.imputation_repeats),
            ],
            cwd=args.repo_root,
            env=env,
            log_path=run_root / "imputation.log",
        )

    return {
        "name": name,
        "description": description,
        "patch": patch or {},
        "gpu": gpu_id,
        "clustering_table": clustering_table,
        "imputation_table": imputation_table,
    }


def _build_markdown(
    out_path: Path,
    args: argparse.Namespace,
    summary_rows: List[Dict[str, Any]],
    ari_rows: List[Dict[str, Any]],
    mse_rows: List[Dict[str, Any]],
) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Aggregate dataset delta summaries per ablation.
    ari_by_ablation: Dict[str, List[Tuple[str, float, float]]] = {}
    for row in ari_rows:
        name = str(row["ablation"])
        ari_by_ablation.setdefault(name, []).append(
            (str(row["dataset"]), float(row["ari"]), float(row["ari_delta_vs_baseline"]))
        )

    mse_by_ablation: Dict[str, List[Tuple[str, float, float, float, float]]] = {}
    for row in mse_rows:
        name = str(row["ablation"])
        mse_by_ablation.setdefault(name, []).append(
            (
                str(row["dataset"]),
                float(row["mse"]),
                float(row["mse_delta_vs_baseline"]),
                float(row["mse_biozero"]),
                float(row["mse_biozero_delta_vs_baseline"]),
            )
        )

    lines: List[str] = []
    lines.append("# MaskClass Ablation Study")
    lines.append("")
    lines.append(f"Generated: {ts}")
    lines.append("")
    lines.append("## Experimental Setup")
    lines.append("")
    lines.append(f"- Clustering input: `{args.clustering_input}`")
    lines.append(f"- Imputation input: `{args.imputation_input}`")
    lines.append(f"- Clustering repeats: `{args.clustering_repeats}`")
    lines.append(f"- Imputation repeats: `{args.imputation_repeats}`")
    lines.append(f"- GPUs used: `{','.join(args.gpus)}`")
    lines.append("")
    lines.append("## Overall Ablation Effects")
    lines.append("")
    lines.append(
        "| Ablation | Description | Mean ARI | ΔARI | Mean MSE | ΔMSE | Mean Biozero MSE | ΔBiozero MSE | all MSE<1 | all Biozero MSE<0.2 |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|:---:|:---:|")
    for row in summary_rows:
        lines.append(
            "| {ablation} | {desc} | {ari} | {dari} | {mse} | {dmse} | {bio} | {dbio} | {mse_ok} | {bio_ok} |".format(
                ablation=row["ablation"],
                desc=row["description"],
                ari=_fmt(float(row["ari_mean"])),
                dari=_fmt(float(row["ari_mean_delta_vs_baseline"])),
                mse=_fmt(float(row["mse_mean"])),
                dmse=_fmt(float(row["mse_mean_delta_vs_baseline"])),
                bio=_fmt(float(row["mse_biozero_mean"])),
                dbio=_fmt(float(row["mse_biozero_mean_delta_vs_baseline"])),
                mse_ok="Y" if str(row["all_mse_lt_1"]).lower() == "true" else "N",
                bio_ok="Y" if str(row["all_mse_biozero_lt_0p2"]).lower() == "true" else "N",
            )
        )

    lines.append("")
    lines.append("## Dataset-level ARI Impact (vs baseline)")
    lines.append("")
    lines.append("| Ablation | Dataset | ARI | ΔARI |")
    lines.append("|---|---|---:|---:|")
    for name in [r["ablation"] for r in summary_rows]:
        for ds, ari, d in sorted(ari_by_ablation.get(name, []), key=lambda x: x[0]):
            lines.append(f"| {name} | {ds} | {_fmt(ari)} | {_fmt(d)} |")

    lines.append("")
    lines.append("## Dataset-level MSE Impact (vs baseline)")
    lines.append("")
    lines.append("| Ablation | Dataset | MSE | ΔMSE | Biozero MSE | ΔBiozero MSE |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for name in [r["ablation"] for r in summary_rows]:
        for ds, mse, dmse, mb, dmb in sorted(mse_by_ablation.get(name, []), key=lambda x: x[0]):
            lines.append(
                f"| {name} | {ds} | {_fmt(mse)} | {_fmt(dmse)} | {_fmt(mb)} | {_fmt(dmb)} |"
            )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ablation study for maskclass balanced_mse.")
    parser.add_argument("--clustering-input", default="datasets", help="Input path for run_clustering.py")
    parser.add_argument(
        "--imputation-input",
        default="synthetic_datasets/rds_splat_output/cells_100",
        help="Input path for run_imputation.py",
    )
    parser.add_argument("--out-dir", default="results_ablation_maskclass", help="Output directory")
    parser.add_argument("--python", default=sys.executable, help="Python executable to run benchmark scripts")
    parser.add_argument("--clustering-repeats", type=int, default=3)
    parser.add_argument("--imputation-repeats", type=int, default=5)
    parser.add_argument("--cpu-threads", type=int, default=8, help="CPU thread cap per benchmark subprocess.")
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="Force CPU execution by clearing CUDA_VISIBLE_DEVICES for benchmark subprocesses.",
    )
    parser.add_argument(
        "--gpus",
        default="0",
        help="Comma-separated GPU IDs used by worker pool, e.g. 0,1,2,3",
    )
    parser.add_argument(
        "--only",
        default="",
        help="Comma-separated subset of ablations to run (include 'baseline' explicitly if desired).",
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Skip execution if both output tables already exist for an ablation.",
    )
    args = parser.parse_args()

    args.repo_root = Path(__file__).resolve().parent
    args.out_dir = str(Path(args.out_dir).resolve())
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    args.gpus = [g.strip() for g in str(args.gpus).split(",") if g.strip()]
    if not args.gpus:
        raise SystemExit("No GPU IDs provided. Use --gpus.")

    defaults = _load_maskclass_defaults()

    specs = _default_ablations()
    selected_names = {x.strip() for x in args.only.split(",") if x.strip()}

    jobs: List[Tuple[str, str, Optional[Dict[str, Dict[str, Any]]]]] = [
        ("baseline", "Default maskclass configuration.", None)
    ]
    for spec in specs:
        if selected_names and spec.name not in selected_names:
            continue
        jobs.append((spec.name, spec.description, spec.patch))

    if selected_names and "baseline" in selected_names and all(j[0] != "baseline" for j in jobs):
        jobs.insert(0, ("baseline", "Default maskclass configuration.", None))

    if not jobs:
        raise SystemExit("No ablations selected.")

    print(f"Running {len(jobs)} ablation jobs with GPUs: {', '.join(args.gpus)}")
    print(f"Output root: {out_root}")

    gpu_pool: Queue[str] = Queue()
    for g in args.gpus:
        gpu_pool.put(g)

    results: Dict[str, Dict[str, Any]] = {}

    def _worker(job: Tuple[str, str, Optional[Dict[str, Dict[str, Any]]]]) -> Dict[str, Any]:
        name, desc, patch = job
        gpu = gpu_pool.get()
        try:
            eff_patch = {} if patch is None else deepcopy(_merge_config(defaults, patch))
            if patch is None:
                eff_patch = {}
            print(f"[START] {name} on GPU {gpu}")
            out = _execute_job(name, desc, patch, args, gpu, out_root)
            out["effective_config"] = eff_patch
            print(f"[DONE ] {name} on GPU {gpu}")
            return out
        finally:
            gpu_pool.put(gpu)

    max_workers = max(1, len(args.gpus))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        fut_to_name = {ex.submit(_worker, job): job[0] for job in jobs}
        for fut in as_completed(fut_to_name):
            name = fut_to_name[fut]
            try:
                out = fut.result()
                results[name] = out
            except Exception as exc:
                print(f"[FAIL] {name}: {exc}")
                return 2

    if "baseline" not in results:
        raise SystemExit("Baseline did not complete; cannot compute deltas.")

    # Parse metrics.
    ari_by_ablation: Dict[str, Dict[str, float]] = {}
    mse_by_ablation: Dict[str, Dict[str, float]] = {}
    mse_bio_by_ablation: Dict[str, Dict[str, float]] = {}

    for name, out in results.items():
        crows = _read_tsv(Path(out["clustering_table"]))
        irows = _read_tsv(Path(out["imputation_table"]))

        ari_map = {row["dataset"]: _parse_float(row.get("ARI", "")) for row in crows}
        mse_map = {row["dataset"]: _parse_float(row.get("mse", "")) for row in irows}
        mse_bio_map = {row["dataset"]: _parse_float(row.get("mse_biozero", "")) for row in irows}

        ari_by_ablation[name] = ari_map
        mse_by_ablation[name] = mse_map
        mse_bio_by_ablation[name] = mse_bio_map

    baseline_ari = ari_by_ablation["baseline"]
    baseline_mse = mse_by_ablation["baseline"]
    baseline_mse_bio = mse_bio_by_ablation["baseline"]

    # Output tables.
    summary_rows: List[Dict[str, Any]] = []
    ari_rows: List[Dict[str, Any]] = []
    mse_rows: List[Dict[str, Any]] = []

    ordered_names = [j[0] for j in jobs]
    for name in ordered_names:
        desc = results[name]["description"]
        patch = results[name]["patch"]

        ari_map = ari_by_ablation[name]
        mse_map = mse_by_ablation[name]
        mse_bio_map = mse_bio_by_ablation[name]

        for ds, val in sorted(ari_map.items()):
            ari_rows.append(
                {
                    "ablation": name,
                    "description": desc,
                    "dataset": ds,
                    "ari": val,
                    "ari_baseline": baseline_ari.get(ds, float("nan")),
                    "ari_delta_vs_baseline": val - baseline_ari.get(ds, float("nan")),
                }
            )

        for ds, val in sorted(mse_map.items()):
            mb = mse_bio_map.get(ds, float("nan"))
            mse_rows.append(
                {
                    "ablation": name,
                    "description": desc,
                    "dataset": ds,
                    "mse": val,
                    "mse_baseline": baseline_mse.get(ds, float("nan")),
                    "mse_delta_vs_baseline": val - baseline_mse.get(ds, float("nan")),
                    "mse_biozero": mb,
                    "mse_biozero_baseline": baseline_mse_bio.get(ds, float("nan")),
                    "mse_biozero_delta_vs_baseline": mb - baseline_mse_bio.get(ds, float("nan")),
                }
            )

        ari_mean = _mean(ari_map.values())
        mse_mean = _mean(mse_map.values())
        mse_bio_mean = _mean(mse_bio_map.values())

        base_ari_mean = _mean(baseline_ari.values())
        base_mse_mean = _mean(baseline_mse.values())
        base_mse_bio_mean = _mean(baseline_mse_bio.values())

        all_mse_lt_1 = all((v < 1.0) for v in mse_map.values() if math.isfinite(v))
        all_mse_bio_lt = all((v < 0.2) for v in mse_bio_map.values() if math.isfinite(v))

        summary_rows.append(
            {
                "ablation": name,
                "description": desc,
                "patch_json": json.dumps(patch or {}, separators=(",", ":")),
                "ari_mean": ari_mean,
                "ari_mean_delta_vs_baseline": ari_mean - base_ari_mean,
                "mse_mean": mse_mean,
                "mse_mean_delta_vs_baseline": mse_mean - base_mse_mean,
                "mse_biozero_mean": mse_bio_mean,
                "mse_biozero_mean_delta_vs_baseline": mse_bio_mean - base_mse_bio_mean,
                "all_mse_lt_1": all_mse_lt_1,
                "all_mse_biozero_lt_0p2": all_mse_bio_lt,
            }
        )

    summary_path = out_root / "ablation_summary.tsv"
    ari_path = out_root / "ablation_ari_by_dataset.tsv"
    mse_path = out_root / "ablation_mse_by_dataset.tsv"
    report_path = out_root / "ablation_report.md"
    config_path = out_root / "effective_configs.json"

    _write_tsv(
        summary_path,
        summary_rows,
        [
            "ablation",
            "description",
            "patch_json",
            "ari_mean",
            "ari_mean_delta_vs_baseline",
            "mse_mean",
            "mse_mean_delta_vs_baseline",
            "mse_biozero_mean",
            "mse_biozero_mean_delta_vs_baseline",
            "all_mse_lt_1",
            "all_mse_biozero_lt_0p2",
        ],
    )
    _write_tsv(
        ari_path,
        ari_rows,
        [
            "ablation",
            "description",
            "dataset",
            "ari",
            "ari_baseline",
            "ari_delta_vs_baseline",
        ],
    )
    _write_tsv(
        mse_path,
        mse_rows,
        [
            "ablation",
            "description",
            "dataset",
            "mse",
            "mse_baseline",
            "mse_delta_vs_baseline",
            "mse_biozero",
            "mse_biozero_baseline",
            "mse_biozero_delta_vs_baseline",
        ],
    )

    _build_markdown(report_path, args, summary_rows, ari_rows, mse_rows)

    eff_cfg = {name: results[name].get("effective_config", {}) for name in ordered_names}
    config_path.write_text(json.dumps(eff_cfg, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("\nAblation study completed.")
    print(f"- Summary: {summary_path}")
    print(f"- ARI by dataset: {ari_path}")
    print(f"- MSE by dataset: {mse_path}")
    print(f"- Report: {report_path}")
    print(f"- Effective configs: {config_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
