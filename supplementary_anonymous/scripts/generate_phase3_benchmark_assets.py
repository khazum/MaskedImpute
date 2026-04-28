#!/usr/bin/env python3
"""Generate Phase 3 synthetic benchmark tables and figures."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon


OUT_DIR = Path("paper/generated")
FIG_DIR = Path("paper/figures")

METHODS = [
    ("MaskImpute", "results_imputation_py/balanced_mse/test_phase2_retuned", "per_dataset"),
    ("DCA", "results_imputation_py/dca/test", "mse_table"),
    ("scVI", "results_imputation_py/scvi/test", "mse_table"),
    ("ALRA", "results_imputation_py/alra/test", "mse_table"),
    ("MAGIC", "results_imputation_py/magic/test", "mse_table"),
    ("AutoClass", "results_imputation_py/autoclass/test", "mse_table"),
    ("ccImpute", "results_imputation_r/ccimpute/test", "mse_table"),
    ("SAVER", "results_imputation_r/saver/test", "mse_table"),
    ("Baseline", "results_imputation_r/baseline/test", "mse_table"),
]

METRICS = [
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

CORE_METRICS = ["mse", "mse_dropout", "mse_biozero", "mae", "gnrmse"]
MARKER_METRICS = ["mse_marker", "mae_marker", "gnrmse_marker"]
FIG_METRICS = [
    "mse",
    "mse_dropout",
    "mse_biozero",
    "mae",
    "mae_dropout",
    "mae_biozero",
    "gnrmse",
    "mse_marker",
]

METRIC_LABELS = {
    "mse": "MSE",
    "mse_dropout": "Dropout-MSE",
    "mse_biozero": "Biozero-MSE",
    "mse_non_zero": "Nonzero-MSE",
    "mse_marker": "Marker-MSE",
    "mae": "MAE",
    "mae_dropout": "Dropout-MAE",
    "mae_biozero": "Biozero-MAE",
    "mae_non_zero": "Nonzero-MAE",
    "mae_marker": "Marker-MAE",
    "gnrmse": "gNRMSE",
    "gnrmse_marker": "Marker-gNRMSE",
    "corr_err": "CorrErr",
    "runtime_sec": "Runtime (s)",
}


def _to_float(value: object) -> float:
    if isinstance(value, str) and value.strip().upper() in {"", "NA", "NAN", "NULL"}:
        return float("nan")
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else float("nan")


def _std(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")


def _tex_escape(text: str) -> str:
    return str(text).replace("_", "\\_")


def _fmt_num(value: float, digits: int = 4) -> str:
    if not math.isfinite(value):
        return "--"
    return f"{value:.{digits}f}"


def _fmt_mean_std(mean: float, std: float, *, bold: bool = False) -> str:
    body = _fmt_num(mean)
    if math.isfinite(std):
        body = f"{body} $\\pm$ {_fmt_num(std)}"
    return f"\\textbf{{{body}}}" if bold else body


def _read_tsv_row(path: Path) -> Dict[str, object]:
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    if not rows:
        raise RuntimeError(f"No rows in {path}")
    return rows[0]


def load_method_rows(root: Path, source_kind: str) -> Dict[str, Dict[str, object]]:
    rows: Dict[str, Dict[str, object]] = {}
    if source_kind == "per_dataset":
        files = sorted(root.glob("*/per_dataset.tsv"))
    elif source_kind == "mse_table":
        files = sorted(root.glob("*/*_mse_table.tsv"))
    else:
        raise ValueError(source_kind)

    for path in files:
        row = _read_tsv_row(path)
        dataset = str(row.get("dataset") or path.parent.name)
        error_value = str(row.get("error", "")).strip()
        if error_value and error_value.upper() not in {"NA", "NAN", "NULL"}:
            raise RuntimeError(f"{dataset}: error row in {path}: {row.get('error')}")
        parsed: Dict[str, object] = {"dataset": dataset}
        for key, value in row.items():
            if key == "dataset":
                continue
            parsed[key] = _to_float(value)
        rows[dataset] = parsed
    if not rows:
        raise RuntimeError(f"No result rows loaded from {root}")
    return rows


def load_all_results() -> Dict[str, Dict[str, Dict[str, object]]]:
    all_rows = {}
    for method, root, kind in METHODS:
        rows = load_method_rows(Path(root), kind)
        all_rows[method] = rows
    scenarios = sorted(set.intersection(*(set(rows) for rows in all_rows.values())))
    if len(scenarios) != 13:
        raise RuntimeError(f"Expected 13 common scenarios, found {len(scenarios)}: {scenarios}")
    # Drop any extra scenarios outside the common synthetic test split.
    return {
        method: {scenario: rows[scenario] for scenario in scenarios}
        for method, rows in all_rows.items()
    }


def summarize(results: Dict[str, Dict[str, Dict[str, object]]]) -> Dict[str, object]:
    method_summary: Dict[str, Dict[str, float]] = {}
    for method, rows in results.items():
        summary = {"n_datasets": len(rows)}
        for metric in METRICS:
            vals = [_to_float(row.get(metric)) for row in rows.values()]
            summary[metric] = _mean(vals)
            summary[f"{metric}_scenario_std"] = _std(vals)
        method_summary[method] = summary

    scenarios = sorted(next(iter(results.values())).keys())
    win_counts: Dict[str, Dict[str, int]] = {}
    for metric in METRICS:
        counts = defaultdict(int)
        for scenario in scenarios:
            vals = {
                method: _to_float(rows[scenario].get(metric))
                for method, rows in results.items()
            }
            vals = {method: value for method, value in vals.items() if math.isfinite(value)}
            if not vals:
                continue
            best = min(vals.values())
            for method, value in vals.items():
                if abs(value - best) <= 1e-12:
                    counts[method] += 1
        if counts:
            win_counts[metric] = dict(counts)

    relative_to_dca = {}
    for metric in METRICS:
        mask = method_summary["MaskImpute"][metric]
        dca = method_summary["DCA"][metric]
        relative_to_dca[metric] = (mask - dca) / dca if math.isfinite(mask) and dca else float("nan")

    significance = {}
    for metric in CORE_METRICS + MARKER_METRICS:
        mask_vals = []
        dca_vals = []
        for scenario in scenarios:
            mask_value = _to_float(results["MaskImpute"][scenario].get(metric))
            dca_value = _to_float(results["DCA"][scenario].get(metric))
            if math.isfinite(mask_value) and math.isfinite(dca_value):
                mask_vals.append(mask_value)
                dca_vals.append(dca_value)
        if len(mask_vals) >= 2:
            try:
                two_sided = wilcoxon(mask_vals, dca_vals, alternative="two-sided").pvalue
                less = wilcoxon(mask_vals, dca_vals, alternative="less").pvalue
            except ValueError:
                two_sided = float("nan")
                less = float("nan")
            significance[metric] = {
                "n": len(mask_vals),
                "wilcoxon_two_sided_p": float(two_sided),
                "wilcoxon_maskimpute_less_p": float(less),
            }

    return {
        "method_summary": method_summary,
        "win_counts": win_counts,
        "relative_to_dca": relative_to_dca,
        "significance_vs_dca": significance,
        "scenarios": scenarios,
    }


def write_summary_table(path: Path, summary: Dict[str, object], metrics: List[str]) -> None:
    method_summary = summary["method_summary"]
    best_by_metric = {
        metric: min(
            _to_float(method_summary[method][metric])
            for method, _, _ in METHODS
            if math.isfinite(_to_float(method_summary[method][metric]))
        )
        for metric in metrics
    }
    align = "l" + "r" * len(metrics)
    lines = [f"\\begin{{tabular}}{{@{{}}{align}@{{}}}}", "\\toprule"]
    lines.append("Method & " + " & ".join(METRIC_LABELS[m] for m in metrics) + " \\\\")
    lines.append("\\midrule")
    for method, _, _ in METHODS:
        cells = [_tex_escape(method)]
        for metric in metrics:
            mean = _to_float(method_summary[method][metric])
            std = _to_float(method_summary[method][f"{metric}_scenario_std"])
            bold = math.isfinite(mean) and abs(mean - best_by_metric[metric]) <= 1e-12
            cells.append(_fmt_mean_std(mean, std, bold=bold))
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def write_by_scenario_table(
    path: Path,
    results: Dict[str, Dict[str, Dict[str, object]]],
    metric: str,
) -> None:
    methods = [m[0] for m in METHODS]
    scenarios = sorted(next(iter(results.values())).keys())
    align = "l" + "r" * len(methods)
    lines = [f"\\begin{{tabular}}{{@{{}}{align}@{{}}}}", "\\toprule"]
    lines.append("Scenario & " + " & ".join(_tex_escape(m) for m in methods) + " \\\\")
    lines.append("\\midrule")
    for scenario in scenarios:
        cells = [_tex_escape(scenario)]
        vals = {
            method: _to_float(results[method][scenario].get(metric))
            for method in methods
        }
        finite_vals = [v for v in vals.values() if math.isfinite(v)]
        best = min(finite_vals) if finite_vals else float("nan")
        for method in methods:
            mean = vals[method]
            std = _to_float(results[method][scenario].get(f"{metric}_std"))
            bold = math.isfinite(mean) and math.isfinite(best) and abs(mean - best) <= 1e-12
            cells.append(_fmt_mean_std(mean, std, bold=bold))
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def write_wins_table(path: Path, summary: Dict[str, object]) -> None:
    lines = ["\\begin{tabular}{@{}ll@{}}", "\\toprule", "Metric & Scenario-level wins \\\\", "\\midrule"]
    for metric in CORE_METRICS + ["mae_biozero"] + MARKER_METRICS:
        counts = summary["win_counts"].get(metric, {})
        total = sum(counts.values())
        parts = [f"{_tex_escape(method)}: {count}/{total}" for method, count in sorted(counts.items(), key=lambda x: (-x[1], x[0]))]
        lines.append(f"{METRIC_LABELS[metric]} & {'; '.join(parts)} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def write_significance_table(path: Path, summary: Dict[str, object]) -> None:
    lines = [
        "\\begin{tabular}{@{}lrr@{}}",
        "\\toprule",
        "Metric & Wilcoxon two-sided $p$ & Wilcoxon MaskImpute $<$ DCA $p$ \\\\",
        "\\midrule",
    ]
    for metric in CORE_METRICS + MARKER_METRICS:
        row = summary["significance_vs_dca"].get(metric, {})
        lines.append(
            f"{METRIC_LABELS[metric]} & {_fmt_num(_to_float(row.get('wilcoxon_two_sided_p')), 4)} & "
            f"{_fmt_num(_to_float(row.get('wilcoxon_maskimpute_less_p')), 4)} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def write_metric_figure(path_pdf: Path, path_png: Path, summary: Dict[str, object]) -> None:
    methods = [m[0] for m in METHODS]
    colors = {
        "MaskImpute": "#2f6f73",
        "DCA": "#c66b3d",
        "scVI": "#5f63b8",
        "ALRA": "#2d9a72",
        "MAGIC": "#7d8f3f",
        "AutoClass": "#8b6f47",
        "ccImpute": "#4d6b9f",
        "SAVER": "#9f5577",
        "Baseline": "#777777",
    }
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.2), constrained_layout=True)
    for ax, metric in zip(axes.flat, FIG_METRICS):
        means = np.array([summary["method_summary"][method][metric] for method in methods], dtype=float)
        stds = np.array([summary["method_summary"][method][f"{metric}_scenario_std"] for method in methods], dtype=float)
        stds = np.nan_to_num(stds, nan=0.0, posinf=0.0, neginf=0.0)
        lower = np.minimum(stds, np.maximum(means, 0.0))
        upper = stds
        x = np.arange(len(methods))
        ax.bar(x, means, color=[colors[m] for m in methods], edgecolor="black", linewidth=0.35)
        ax.errorbar(x, means, yerr=np.vstack([lower, upper]), fmt="none", ecolor="#252525", elinewidth=0.8, capsize=2)
        ax.set_title(METRIC_LABELS[metric])
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha="right")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", color="#d8d8d8", linewidth=0.5)
        ax.set_axisbelow(True)
    fig.suptitle("Synthetic test split benchmark metrics", fontsize=15, fontweight="bold")
    fig.savefig(path_pdf)
    fig.savefig(path_png, dpi=220)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    results = load_all_results()
    summary = summarize(results)

    write_summary_table(OUT_DIR / "benchmark_core_table.tex", summary, CORE_METRICS)
    write_summary_table(OUT_DIR / "benchmark_marker_table.tex", summary, MARKER_METRICS)
    write_wins_table(OUT_DIR / "benchmark_wins_table.tex", summary)
    write_significance_table(OUT_DIR / "benchmark_significance_table.tex", summary)

    by_scenario = {
        "mse": "benchmark_mse_by_scenario.tex",
        "mse_dropout": "benchmark_mse_dropout_by_scenario.tex",
        "mse_biozero": "benchmark_mse_biozero_by_scenario.tex",
        "mse_marker": "benchmark_mse_marker_by_scenario.tex",
        "mae": "benchmark_mae_by_scenario.tex",
        "mae_dropout": "benchmark_mae_dropout_by_scenario.tex",
        "mae_biozero": "benchmark_mae_biozero_by_scenario.tex",
        "mae_marker": "benchmark_mae_marker_by_scenario.tex",
        "gnrmse": "benchmark_gnrmse_by_scenario.tex",
        "gnrmse_marker": "benchmark_gnrmse_marker_by_scenario.tex",
        "corr_err": "benchmark_corr_err_by_scenario.tex",
        "runtime_sec": "benchmark_runtime_sec_by_scenario.tex",
    }
    for metric, filename in by_scenario.items():
        write_by_scenario_table(OUT_DIR / filename, results, metric)

    write_metric_figure(FIG_DIR / "mse_5000_bars.pdf", FIG_DIR / "mse_5000_bars.png", summary)
    (OUT_DIR / "benchmark_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary["method_summary"]["MaskImpute"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
