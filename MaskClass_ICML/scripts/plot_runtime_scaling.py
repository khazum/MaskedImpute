#!/usr/bin/env python3
"""Generate runtime scaling plot from imputation results."""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

METHOD_LABELS = {
    "baseline": "Baseline",
    "saver": "SAVER",
    "ccimpute": "ccImpute",
    "magic": "MAGIC",
    "dca": "DCA",
    "autoclass": "AutoClass",
    "balanced_mse": "MaskClass",
}

METHOD_ORDER = [
    "baseline",
    "saver",
    "ccimpute",
    "magic",
    "dca",
    "autoclass",
    "balanced_mse",
]


def _collect_rows(base_dir: Path, method: str) -> list[dict]:
    rows: list[dict] = []
    for path in base_dir.glob(f"{method}/cells_*/*_mse_table.tsv"):
        match = re.search(r"cells_(\d+)", str(path))
        if not match:
            continue
        size = int(match.group(1))
        df = pd.read_csv(path, sep="\t")
        if "runtime_sec" not in df.columns:
            continue
        runtimes = pd.to_numeric(df["runtime_sec"], errors="coerce")
        for rt in runtimes:
            if np.isfinite(rt):
                rows.append({"method": method, "size": size, "runtime": float(rt)})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-py", default="results_imputation_py")
    parser.add_argument("--results-r", default="results_imputation_r")
    parser.add_argument("--out-pdf", default="MaskClass_ICML/figures/runtime_scaling.pdf")
    parser.add_argument("--out-png", default="MaskClass_ICML/figures/runtime_scaling.png")
    args = parser.parse_args()

    rows: list[dict] = []
    rows.extend(_collect_rows(Path(args.results_py), "magic"))
    rows.extend(_collect_rows(Path(args.results_py), "dca"))
    rows.extend(_collect_rows(Path(args.results_py), "autoclass"))
    rows.extend(_collect_rows(Path(args.results_py), "balanced_mse"))
    rows.extend(_collect_rows(Path(args.results_r), "baseline"))
    rows.extend(_collect_rows(Path(args.results_r), "saver"))
    rows.extend(_collect_rows(Path(args.results_r), "ccimpute"))

    if not rows:
        raise SystemExit("No runtime data found.")

    df = pd.DataFrame(rows)
    summary = (
        df.groupby(["method", "size"])["runtime"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for method in METHOD_ORDER:
        sub = summary[summary["method"] == method].sort_values("size")
        if sub.empty:
            continue
        label = METHOD_LABELS.get(method, method)
        ax.errorbar(
            sub["size"],
            sub["mean"],
            yerr=sub["std"].fillna(0.0),
            marker="o",
            linewidth=1.6,
            markersize=4,
            capsize=2,
            label=label,
        )

    ax.set_xlabel("Number of cells (N)")
    ax.set_ylabel("Runtime per dataset (seconds)")
    ax.set_title("Runtime scaling on synthetic datasets")
    ax.set_xscale("linear")
    ax.set_yscale("linear")
    ax.grid(True, which="major", axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.set_xticks(sorted(df["size"].unique()))
    ax.set_xticklabels([f"{int(x/1000)}k" for x in sorted(df["size"].unique())])
    ax.legend(ncol=2, fontsize=8, frameon=False)

    out_pdf = Path(args.out_pdf)
    out_png = Path(args.out_png)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=200)


if __name__ == "__main__":
    main()
