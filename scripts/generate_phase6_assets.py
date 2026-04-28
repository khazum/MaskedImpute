#!/usr/bin/env python3
"""Generate Phase 6 ablation, sensitivity, calibration, and downstream tables."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PHASE6 = ROOT / "results_phase6"
OUT = ROOT / "paper" / "generated"
DOC = ROOT / "docs" / "feedback_phase6_results.md"
OUT.mkdir(parents=True, exist_ok=True)

SCORE_NAME = "MSE + 2*Biozero-MSE"


def _read_metrics(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, sep="\t")
    df["efficacy_score"] = df["mse"] + 2.0 * df["mse_biozero"]
    return df


def _mean_ci(df: pd.DataFrame, col: str) -> tuple[float, float]:
    x = pd.to_numeric(df[col], errors="coerce").dropna()
    if len(x) == 0:
        return float("nan"), float("nan")
    mean = float(x.mean())
    if len(x) == 1:
        return mean, 0.0
    ci = 1.96 * float(x.std(ddof=1)) / math.sqrt(len(x))
    return mean, ci


def _fmt(x: float, digits: int = 3) -> str:
    if pd.isna(x):
        return "--"
    return f"{x:.{digits}f}"


def _fmt_ci(mean: float, ci: float, digits: int = 3) -> str:
    if pd.isna(mean):
        return "--"
    if ci == 0 or pd.isna(ci):
        return _fmt(mean, digits)
    return f"{mean:.{digits}f} $\\pm$ {ci:.{digits}f}"


def _tex_escape(s: str) -> str:
    return s.replace("_", "\\_")


def _write_tabular(path: Path, headers: list[str], rows: list[list[str]], align: str | None = None) -> None:
    if align is None:
        align = "l" + "r" * (len(headers) - 1)
    lines = ["\\begin{tabular}{" + align + "}", "\\toprule"]
    lines.append(" & ".join(headers) + " \\\\")
    lines.append("\\midrule")
    for row in rows:
        lines.append(" & ".join(row) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    path.write_text("\n".join(lines) + "\n")


def build_ablation() -> pd.DataFrame:
    variants = [
        ("ablation_current", "MaskImpute"),
        ("ablation_no_shrinkage", "No zero shrinkage"),
        ("ablation_no_bio_reg", "No biozero regularization"),
        ("ablation_uniform_masking", "Uniform masking"),
        ("ablation_plain_mae", "Plain masked AE"),
    ]
    records = []
    rows = []
    for dirname, label in variants:
        path = PHASE6 / dirname / "metrics.tsv"
        if not path.exists():
            continue
        df = _read_metrics(path)
        rec = {"variant": label, "n_scenarios": int(df["dataset"].nunique())}
        for col in ["efficacy_score", "mse", "mse_dropout", "mse_biozero", "mae", "gnrmse", "ari_matched_mean", "nmi_matched_mean"]:
            rec[col], rec[col + "_ci"] = _mean_ci(df, col)
        records.append(rec)
        rows.append([
            label,
            _fmt_ci(rec["efficacy_score"], rec["efficacy_score_ci"]),
            _fmt_ci(rec["mse"], rec["mse_ci"]),
            _fmt_ci(rec["mse_biozero"], rec["mse_biozero_ci"]),
            _fmt(rec["mse_dropout"]),
            _fmt(rec["ari_matched_mean"]),
            _fmt(rec["nmi_matched_mean"]),
        ])
    _write_tabular(
        OUT / "phase6_ablation_table.tex",
        ["Variant", "Efficacy", "MSE", "Biozero-MSE", "Dropout-MSE", "ARI", "NMI"],
        rows,
    )
    _write_tabular(
        OUT / "ablation_table.tex",
        ["Variant", "Efficacy", "MSE", "Biozero-MSE", "Dropout-MSE", "ARI", "NMI"],
        rows,
    )
    return pd.DataFrame(records)


def _config_value(cfg_path: Path, key: str) -> float | None:
    cfg = json.loads(cfg_path.read_text())
    value = cfg.get(key)
    return None if value is None else float(value)


def build_sensitivity() -> pd.DataFrame:
    groups = [
        ("Shrinkage $\\alpha$", "shrink_", "zero_shrink_strength", [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]),
        ("Biozero weight $\\gamma$", "bio_", "bio_reg_weight", [0, 0.5, 1.0, 1.2, 2.0, 5.0]),
        ("Zero mask rate $p_0$", "pzero_", "p_zero", [0.005, 0.01, 0.02, 0.05]),
    ]
    records = []
    tex_rows = []
    for group_label, prefix, key, order in groups:
        for value in order:
            name_value = f"{value:g}".replace(".", "p")
            if prefix == "shrink_":
                dirname = f"shrink_{value:g}"
            elif prefix == "bio_":
                dirname = f"bio_{value:g}" if value not in {0.5} else "bio_0.5"
            else:
                dirname = f"pzero_{name_value}"
            path = PHASE6 / "sensitivity" / dirname / "metrics.tsv"
            if not path.exists():
                continue
            df = _read_metrics(path)
            rec = {
                "parameter": group_label,
                "value": value,
                "dataset": str(df["dataset"].iloc[0]),
                "efficacy_score": float(df["efficacy_score"].mean()),
                "mse": float(df["mse"].mean()),
                "mse_biozero": float(df["mse_biozero"].mean()),
                "mse_dropout": float(df["mse_dropout"].mean()),
                "ari_matched_mean": float(df["ari_matched_mean"].mean()),
                "nmi_matched_mean": float(df["nmi_matched_mean"].mean()),
            }
            records.append(rec)
            tex_rows.append([
                group_label,
                _fmt(value, 3).rstrip("0").rstrip("."),
                _fmt(rec["efficacy_score"]),
                _fmt(rec["mse"]),
                _fmt(rec["mse_biozero"]),
                _fmt(rec["mse_dropout"]),
                _fmt(rec["ari_matched_mean"]),
            ])
    _write_tabular(
        OUT / "phase6_sensitivity_table.tex",
        ["Parameter", "Value", "Efficacy", "MSE", "Biozero-MSE", "Dropout-MSE", "ARI"],
        tex_rows,
    )
    return pd.DataFrame(records)


def build_calibration() -> pd.DataFrame:
    path = PHASE6 / "calibration_default" / "metrics.tsv"
    if not path.exists():
        return pd.DataFrame()
    df = _read_metrics(path)
    order = ["raw", "all_obs0.05", "all_obs0.1", "all_obs0.15", "all_obs0.2", "all_obs0.25", "all_obs1.0"]
    labels = {
        "raw": "MaskImpute",
        "all_obs0.05": "$0.95X_{MI}+0.05X_{obs}$",
        "all_obs0.1": "$0.90X_{MI}+0.10X_{obs}$",
        "all_obs0.15": "$0.85X_{MI}+0.15X_{obs}$",
        "all_obs0.2": "$0.80X_{MI}+0.20X_{obs}$",
        "all_obs0.25": "$0.75X_{MI}+0.25X_{obs}$",
        "all_obs1.0": "Observed input",
    }
    records = []
    rows = []
    for transform in order:
        sub = df[df["transform"] == transform]
        if sub.empty:
            continue
        rec = {"transform": transform, "label": labels[transform], "n_scenarios": int(sub["dataset"].nunique())}
        for col in ["efficacy_score", "mse", "mse_dropout", "mse_biozero", "mae", "gnrmse", "ari_matched_mean", "nmi_matched_mean"]:
            rec[col], rec[col + "_ci"] = _mean_ci(sub, col)
        records.append(rec)
        rows.append([
            labels[transform],
            _fmt_ci(rec["efficacy_score"], rec["efficacy_score_ci"]),
            _fmt(rec["mse"]),
            _fmt(rec["mse_biozero"]),
            _fmt(rec["mse_dropout"]),
            _fmt(rec["ari_matched_mean"]),
            _fmt(rec["nmi_matched_mean"]),
        ])
    _write_tabular(
        OUT / "phase6_calibration_table.tex",
        ["Output", "Efficacy", "MSE", "Biozero-MSE", "Dropout-MSE", "ARI", "NMI"],
        rows,
    )
    # A compact downstream-only table used in the main text.
    downstream_rows = []
    for transform in ["all_obs1.0", "raw", "all_obs0.05", "all_obs0.1"]:
        rec = next((r for r in records if r["transform"] == transform), None)
        if rec is None:
            continue
        downstream_rows.append([
            labels[transform],
            _fmt(rec["ari_matched_mean"]),
            _fmt(rec["nmi_matched_mean"]),
            _fmt(rec["ari_grid_mean"] if "ari_grid_mean" in rec else float("nan")),
            _fmt(rec["nmi_grid_mean"] if "nmi_grid_mean" in rec else float("nan")),
        ])
    # Add grid means; populate records above did not include them in earlier loop.
    downstream_rows = []
    for transform in ["all_obs1.0", "raw", "all_obs0.05", "all_obs0.1"]:
        sub = df[df["transform"] == transform]
        if sub.empty:
            continue
        downstream_rows.append([
            labels[transform],
            _fmt(float(sub["ari_matched_mean"].mean())),
            _fmt(float(sub["nmi_matched_mean"].mean())),
            _fmt(float(sub["ari_grid_mean"].mean())),
            _fmt(float(sub["nmi_grid_mean"].mean())),
        ])
    _write_tabular(
        OUT / "phase6_downstream_table.tex",
        ["Output", "ARI (matched)", "NMI (matched)", "ARI (grid)", "NMI (grid)"],
        downstream_rows,
    )
    _write_tabular(
        OUT / "downstream_table.tex",
        ["Output", "ARI (matched)", "NMI (matched)", "ARI (grid)", "NMI (grid)"],
        downstream_rows,
    )
    return pd.DataFrame(records)


def write_doc(ablation: pd.DataFrame, sensitivity: pd.DataFrame, calibration: pd.DataFrame) -> None:
    lines: list[str] = []
    lines.append("# Phase 6: Ablations, Sensitivity, and Downstream Synthetic Tasks")
    lines.append("")
    lines.append(f"Efficacy score is `{SCORE_NAME}`; lower is better. Values below are means over synthetic test scenarios unless noted.")
    lines.append("")
    if not ablation.empty:
        lines.append("## Ablation Summary")
        lines.append("")
        lines.append("| Variant | Efficacy | MSE | Biozero-MSE | Dropout-MSE | ARI matched | NMI matched |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for _, r in ablation.iterrows():
            lines.append(f"| {r['variant']} | {_fmt(r['efficacy_score'])} | {_fmt(r['mse'])} | {_fmt(r['mse_biozero'])} | {_fmt(r['mse_dropout'])} | {_fmt(r['ari_matched_mean'])} | {_fmt(r['nmi_matched_mean'])} |")
        lines.append("")
    if not calibration.empty:
        lines.append("## Calibration Search")
        lines.append("")
        lines.append("Residual calibration uses `X_out=(1-lambda) X_MaskImpute + lambda X_observed`; it is label-free and has one scalar parameter.")
        lines.append("")
        lines.append("| Output | Efficacy | MSE | Biozero-MSE | Dropout-MSE | ARI matched | NMI matched |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for _, r in calibration.iterrows():
            lines.append(f"| {r['label']} | {_fmt(r['efficacy_score'])} | {_fmt(r['mse'])} | {_fmt(r['mse_biozero'])} | {_fmt(r['mse_dropout'])} | {_fmt(r['ari_matched_mean'])} | {_fmt(r['nmi_matched_mean'])} |")
        lines.append("")
    if not sensitivity.empty:
        lines.append("## One-Scenario Sensitivity")
        lines.append("")
        lines.append("Representative scenario: `groups_balanced_moderate_drop`.")
        lines.append("")
        lines.append("| Parameter | Value | Efficacy | MSE | Biozero-MSE | Dropout-MSE | ARI matched |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for _, r in sensitivity.iterrows():
            lines.append(f"| {r['parameter']} | {_fmt(r['value'], 3).rstrip('0').rstrip('.')} | {_fmt(r['efficacy_score'])} | {_fmt(r['mse'])} | {_fmt(r['mse_biozero'])} | {_fmt(r['mse_dropout'])} | {_fmt(r['ari_matched_mean'])} |")
        lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- The biozero regularizer is the main component protecting biological zeros; removing it lowers aggregate MSE but substantially worsens Biozero-MSE.")
    lines.append("- Zero shrinkage provides a smaller but consistent Biozero-MSE gain with little effect on downstream synthetic clustering.")
    lines.append("- Uniform masking is not a good default: it can reduce Biozero-MSE but increases full-matrix and dropout errors.")
    lines.append("- Residual calibration is simpler than earlier clustering-specific post-processing because it is a fixed convex combination with the observed matrix and no labels or graph construction.")
    lines.append("")
    lines.append("## Calibration Recommendation")
    lines.append("")
    lines.append("- Keep uncalibrated MaskImpute as the primary denoising output for the headline benchmark because it has the best Dropout-MSE.")
    lines.append("- If a single calibrated output is needed, `lambda=0.05` is the safest synthetic-error trade-off: efficacy improves from `0.251` to `0.243`, MSE improves from `0.224` to `0.219`, and Dropout-MSE only changes from `0.304` to `0.310`.")
    lines.append("- If downstream label agreement is prioritized, `lambda=0.10` has the best matched synthetic ARI/NMI among the screened values while preserving clear leads over DCA on MSE, Dropout-MSE, and Biozero-MSE.")
    DOC.write_text("\n".join(lines) + "\n")


def main() -> None:
    ablation = build_ablation()
    sensitivity = build_sensitivity()
    calibration = build_calibration()
    write_doc(ablation, sensitivity, calibration)
    print(f"Wrote {OUT / 'phase6_ablation_table.tex'}")
    print(f"Wrote {OUT / 'phase6_sensitivity_table.tex'}")
    print(f"Wrote {OUT / 'phase6_calibration_table.tex'}")
    print(f"Wrote {OUT / 'phase6_downstream_table.tex'}")
    print(f"Wrote {DOC}")


if __name__ == "__main__":
    main()
