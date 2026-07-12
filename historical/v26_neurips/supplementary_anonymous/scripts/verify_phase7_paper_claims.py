#!/usr/bin/env python3
"""Verify headline numerical claims in paper/main.tex against generated artifacts."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PAPER_FILES = [
    ROOT / "paper/main.tex",
    ROOT / "paper/appendix_synthetic_details.tex",
    ROOT / "paper/checklist.tex",
]
SUMMARY = json.loads((ROOT / "paper/generated/benchmark_summary.json").read_text())
METHODS = SUMMARY["method_summary"]
REAL = pd.read_csv(ROOT / "results_real_data/real_data_clustering_summary.tsv", sep="\t")
PHASE6 = pd.read_csv(ROOT / "results_phase6/calibration_default/summary.tsv", sep="\t")
ABL = {
    p.parent.name: pd.read_csv(p, sep="\t").iloc[0].to_dict()
    for p in sorted((ROOT / "results_phase6").glob("ablation_*/summary.tsv"))
}


def rel_reduction(method: str, baseline: str, metric: str) -> float:
    return 100.0 * (1.0 - float(METHODS[method][metric]) / float(METHODS[baseline][metric]))


def close(value: float, target: float, tol: float) -> bool:
    return abs(value - target) <= tol


def assert_true(cond: bool, message: str) -> None:
    if not cond:
        raise AssertionError(message)


def method_row(dataset: str, method: str):
    sub = REAL[(REAL["dataset"] == dataset) & (REAL["method"] == method)]
    assert_true(not sub.empty, f"missing real-data row: {dataset}/{method}")
    return sub.iloc[0]


def main() -> None:
    checks = []

    # Source, citation, and naming hygiene checks.
    tex = "\n".join(path.read_text(errors="ignore") for path in PAPER_FILES)
    missing_sources = []
    for match in re.finditer(r"\\input\{([^}]+)\}", tex):
        name = match.group(1)
        source = ROOT / "paper" / name
        if not source.suffix:
            source = source.with_suffix(".tex")
        if not source.exists():
            missing_sources.append(str(source.relative_to(ROOT)))
    for match in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex):
        name = match.group(1)
        source = ROOT / "paper" / name
        candidates = [source] if source.suffix else [Path(str(source) + ext) for ext in [".pdf", ".png", ".jpg", ".jpeg"]]
        if not any(path.exists() for path in candidates):
            missing_sources.append(str(source.relative_to(ROOT)))
    assert_true(not missing_sources, "missing paper sources: " + ", ".join(missing_sources))

    cite_keys = set()
    for match in re.finditer(r"\\cite\w*\*?(?:\[[^\]]*\])*\{([^}]*)\}", tex):
        cite_keys.update(k.strip() for k in match.group(1).split(",") if k.strip())
    bib_text = (ROOT / "paper/references.bib").read_text(errors="ignore")
    bib_keys = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)", bib_text))
    missing_cites = sorted(cite_keys - bib_keys)
    assert_true(not missing_cites, "missing bibliography keys: " + ", ".join(missing_cites))

    obsolete = re.compile(r"MaskClass|MaskedImpute|Balanced-MSE|balanced_mse|UAI|uai")
    obsolete_hits = []
    for path in PAPER_FILES + list((ROOT / "paper/generated").glob("*.tex")):
        for line_no, line in enumerate(path.read_text(errors="ignore").splitlines(), start=1):
            if obsolete.search(line):
                obsolete_hits.append(f"{path.relative_to(ROOT)}:{line_no}")
    assert_true(not obsolete_hits, "obsolete paper terms: " + ", ".join(obsolete_hits[:20]))

    # Headline synthetic reductions rounded in the abstract/results.
    checks.append(("MaskImpute MSE vs DCA", rel_reduction("MaskImpute", "DCA", "mse"), 38.0, 1.0))
    checks.append(("MaskImpute MSE vs scVI", rel_reduction("MaskImpute", "scVI", "mse"), 35.0, 1.0))
    checks.append(("MaskImpute Dropout-MSE vs DCA", rel_reduction("MaskImpute", "DCA", "mse_dropout"), 34.0, 1.0))
    checks.append(("MaskImpute Dropout-MSE vs scVI", rel_reduction("MaskImpute", "scVI", "mse_dropout"), 32.0, 1.0))

    assert_true(float(METHODS["MaskImpute"]["mse_biozero"]) < float(METHODS["DCA"]["mse_biozero"]), "MaskImpute Biozero-MSE not lower than DCA")
    assert_true(float(METHODS["MaskImpute"]["mse_biozero"]) < float(METHODS["scVI"]["mse_biozero"]), "MaskImpute Biozero-MSE not lower than scVI")

    for name, value, target, tol in checks:
        assert_true(close(value, target, tol), f"{name}: {value:.2f}% not within {tol} of {target}%")

    # Real-data claims.
    for metric in ["ari_matched_mean", "nmi_matched_mean"]:
        mask = method_row("Baron", "maskimpute")[metric]
        best = REAL[REAL["dataset"] == "Baron"][metric].max()
        assert_true(abs(mask - best) <= 1e-12, f"Baron MaskImpute is not best for {metric}")
    zeisel = REAL[REAL["dataset"] == "Zeisel"]
    assert_true(method_row("Zeisel", "baseline")["ari_matched_mean"] == zeisel["ari_matched_mean"].max(), "Zeisel observed is not strongest by matched ARI")
    impute_zeisel = zeisel[zeisel["method"] != "baseline"]
    assert_true(method_row("Zeisel", "maskimpute")["nmi_matched_mean"] == impute_zeisel["nmi_matched_mean"].max(), "Zeisel MaskImpute not best imputation matched NMI")
    dca_runs = pd.read_csv(ROOT / "results_real_data/real_data_method_runs.tsv", sep="\t")
    dca_status = set(dca_runs[dca_runs["method"] == "dca"]["status"].dropna().str.lower())
    assert_true("timeout" not in dca_status, "DCA timeout present despite paper text")

    # Phase 6 calibration/ablation claims.
    cal = PHASE6.set_index("transform")
    score = cal["mse"] + 2.0 * cal["mse_biozero"]
    assert_true(score["all_obs0.05"] < score["raw"], "lambda=0.05 does not improve efficacy")
    assert_true(cal.loc["raw", "mse_dropout"] < cal.loc["all_obs0.05", "mse_dropout"], "raw no longer has best Dropout-MSE vs lambda=0.05")
    assert_true(cal.loc["all_obs0.1", "ari_matched_mean"] == cal["ari_matched_mean"].max(), "lambda=0.10 is not best matched ARI")
    assert_true(cal.loc["all_obs0.1", "nmi_matched_mean"] == cal["nmi_matched_mean"].max(), "lambda=0.10 is not best matched NMI")
    assert_true(ABL["ablation_no_bio_reg"]["mse_biozero"] > ABL["ablation_current"]["mse_biozero"], "no-bio-reg does not worsen Biozero-MSE")
    assert_true(ABL["ablation_no_shrinkage"]["mse_biozero"] > ABL["ablation_current"]["mse_biozero"], "no-shrinkage does not worsen Biozero-MSE")

    print("Phase 7 claim checks passed")
    print(f"MaskImpute MSE={METHODS['MaskImpute']['mse']:.6f}, Dropout-MSE={METHODS['MaskImpute']['mse_dropout']:.6f}, Biozero-MSE={METHODS['MaskImpute']['mse_biozero']:.6f}")
    print(f"Reductions vs DCA: MSE={rel_reduction('MaskImpute','DCA','mse'):.1f}%, Dropout-MSE={rel_reduction('MaskImpute','DCA','mse_dropout'):.1f}%")
    print(f"Reductions vs scVI: MSE={rel_reduction('MaskImpute','scVI','mse'):.1f}%, Dropout-MSE={rel_reduction('MaskImpute','scVI','mse_dropout'):.1f}%")


if __name__ == "__main__":
    main()
