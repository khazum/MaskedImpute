#!/usr/bin/env python3
"""
Generate ICML-ready bar plots for denoising accuracy on synthetic data (N=5,000).

Reads precomputed TSV tables from results_imputation_{py,r} and outputs a
multi-panel figure (overall MSE / dropout-MSE / biozero-MSE).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _load_tables(repo_root: Path) -> pd.DataFrame:
    files = {
        # Treat balanced_mse as "MaskClass" in the paper (user request).
        "MaskClass": repo_root
        / "results_imputation_py/balanced_mse/cells_5000/balanced_mse_mse_table.tsv",
        "DCA": repo_root / "results_imputation_py/dca/cells_5000/dca_mse_table.tsv",
        "MAGIC": repo_root / "results_imputation_py/magic/cells_5000/magic_mse_table.tsv",
        "AutoClass": repo_root
        / "results_imputation_py/autoclass/cells_5000/autoclass_mse_table.tsv",
        "ccImpute": repo_root
        / "results_imputation_r/ccimpute/cells_5000/ccimpute_mse_table.tsv",
        "Baseline": repo_root
        / "results_imputation_r/baseline/cells_5000/baseline_mse_table.tsv",
    }

    dataset_map = {
        "dataset_2_types_equal": "Sim-Equal",
        "dataset_3_types_unequal": "Sim-Unequal",
        "dataset_4_types_rare": "Sim-Rare",
        "dataset_multibatch_dropout": "Sim-Batch",
    }

    rows = []
    for method, path in files.items():
        df = pd.read_csv(path, sep="\t")
        df["scenario"] = df["dataset"].map(dataset_map)
        for _, r in df.iterrows():
            rows.append(
                {
                    "method": method,
                    "scenario": r["scenario"],
                    "mse": float(r["mse"]),
                    "mse_std": float(r["mse_std"]),
                    "mse_dropout": float(r["mse_dropout"]),
                    "mse_dropout_std": float(r["mse_dropout_std"]),
                    "mse_biozero": float(r["mse_biozero"]),
                    "mse_biozero_std": float(r["mse_biozero_std"]),
                }
            )

    out = pd.DataFrame(rows)
    if out["scenario"].isna().any():
        missing = out.loc[out["scenario"].isna(), ["method"]].drop_duplicates()
        raise ValueError(f"Unmapped scenarios present in the input TSVs: {missing}")
    return out


def main() -> None:
    import matplotlib.pyplot as plt

    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[2]
    paper_root = script_path.parents[1]

    df = _load_tables(repo_root)

    scenario_order = ["Sim-Equal", "Sim-Unequal", "Sim-Rare", "Sim-Batch"]
    # Use explicit line breaks to keep tick labels readable in multi-panel layout.
    scenario_labels = ["Sim-\nEqual", "Sim-\nUnequal", "Sim-\nRare", "Sim-\nBatch"]
    method_order = ["Baseline", "MAGIC", "AutoClass", "ccImpute", "DCA", "MaskClass"]

    # ICML-friendly styling: compact, readable at 10pt.
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        # Older matplotlib; fall back gracefully.
        plt.style.use("seaborn-whitegrid")

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )

    colors = {
        "Baseline": "#7f7f7f",
        "MAGIC": "#1f77b4",
        "AutoClass": "#ff7f0e",
        "ccImpute": "#2ca02c",
        "DCA": "#d62728",
        "MaskClass": "#9467bd",
    }

    metrics = [
        ("mse", "mse_std", "Overall MSE"),
        ("mse_dropout", "mse_dropout_std", "Dropout-MSE"),
        ("mse_biozero", "mse_biozero_std", "Biozero-MSE"),
    ]

    x = np.arange(len(scenario_order))
    width = 0.12
    offsets = np.linspace(
        -width * (len(method_order) - 1) / 2,
        width * (len(method_order) - 1) / 2,
        len(method_order),
    )

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), constrained_layout=False)

    for ax, (mean_col, std_col, title) in zip(axes, metrics):
        for off, method in zip(offsets, method_order):
            sub = (
                df.loc[df["method"] == method, ["scenario", mean_col, std_col]]
                .set_index("scenario")
                .reindex(scenario_order)
            )
            means = sub[mean_col].to_numpy()
            stds = sub[std_col].to_numpy()

            ax.bar(
                x + off,
                means,
                width=width,
                color=colors[method],
                label=method,
                yerr=stds,
                capsize=1.5,
                error_kw={"elinewidth": 0.6, "capthick": 0.6},
            )

        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(scenario_labels)
        ax.tick_params(axis="x", pad=1)
        ax.margins(x=0.01)

        ymax = float(df[mean_col].max())
        ax.set_ylim(0, ymax * 1.15 if ymax > 0 else 1.0)

    # Shared legend.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.08),
    )

    # Reserve bottom margin for multi-line x tick labels + legend.
    fig.tight_layout(rect=(0, 0.12, 1, 1))

    out_pdf = paper_root / "figures" / "mse_5000_bars.pdf"
    out_png = paper_root / "figures" / "mse_5000_bars.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"Wrote: {out_pdf}")
    print(f"Wrote: {out_png}")


if __name__ == "__main__":
    main()
