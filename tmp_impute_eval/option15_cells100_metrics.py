#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tmp_impute_eval.option15_scope5_light import CONFIGS

CELLS100_INPUT = os.getenv("CELLS100_INPUT", "synthetic_datasets/rds_splat_output/cells_100")
CELLS100_OUT = Path(os.getenv("CELLS100_OUT", "tmp_impute_eval/options_scope_dynamic_v1_cells100"))
SCOPE5_OUT = Path(os.getenv("OPTIONS_OUT_ROOT", "tmp_impute_eval/options_scope_dynamic_v1"))
BIOZERO_THRESH = float(os.getenv("BIOZERO_THRESH", "0.25"))
CONFIG_FILTER = os.getenv("CONFIG_FILTER", "").strip()


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


def _write_tsv(path: Path, header: List[str], rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in header})


def run_cells100() -> List[Dict[str, object]]:
    CELLS100_OUT.mkdir(parents=True, exist_ok=True)
    cfgs = CONFIGS
    if CONFIG_FILTER:
        pat = re.compile(CONFIG_FILTER)
        cfgs = [cfg for cfg in CONFIGS if pat.search(str(cfg.get("name", "")))]
    if not cfgs:
        raise SystemExit(f"No configs matched CONFIG_FILTER='{CONFIG_FILTER}'")

    rows: List[Dict[str, object]] = []
    for cfg in cfgs:
        option = int(cfg["option"])
        run_name = str(cfg["name"])
        args = [str(x) for x in cfg.get("args", [])]
        out_dir = CELLS100_OUT / run_name
        out_dir.mkdir(parents=True, exist_ok=True)
        summary_path = out_dir / f"masked_imputation_option{option}_summary.tsv"
        if not summary_path.exists():
            cmd = [
                "python",
                f"masked_imputation_option{option}.py",
                CELLS100_INPUT,
                str(out_dir),
                "--device",
                "cuda",
                "--seed",
                "42",
                "--epochs",
                "120",
                "--save-imputed",
                "false",
            ] + args
            print(f"[RUN] {run_name}: {' '.join(cmd)}", flush=True)
            proc = subprocess.run(cmd, capture_output=True, text=True)
            (out_dir / "run.log").write_text(
                proc.stdout + "\n\n===== STDERR =====\n\n" + proc.stderr, encoding="utf-8"
            )
        else:
            print(f"[SKIP] {run_name}: existing cells_100 summary", flush=True)

        srows = _read_tsv(summary_path)
        avg_mse = float("nan")
        avg_biozero = float("nan")
        runtime_sec = float("nan")
        if srows:
            s0 = srows[0]
            avg_mse = _safe_float(s0.get("avg_mse"))
            avg_biozero = _safe_float(s0.get("avg_biozero"))
            runtime_sec = _safe_float(s0.get("runtime_sec"))

        rows.append(
            {
                "run_name": run_name,
                "option": option,
                "cells100_avg_mse": avg_mse,
                "cells100_avg_biozero": avg_biozero,
                "cells100_runtime_sec": runtime_sec,
                "meets_mse_le_1": int(math.isfinite(avg_mse) and avg_mse <= 1.0),
                "meets_biozero_le_thresh": int(
                    math.isfinite(avg_biozero) and avg_biozero <= BIOZERO_THRESH
                ),
                "meets_joint_cells100": int(
                    (math.isfinite(avg_mse) and avg_mse <= 1.0)
                    and (math.isfinite(avg_biozero) and avg_biozero <= BIOZERO_THRESH)
                ),
            }
        )

    _write_tsv(
        CELLS100_OUT / "cells100_metrics.tsv",
        [
            "run_name",
            "option",
            "cells100_avg_mse",
            "cells100_avg_biozero",
            "cells100_runtime_sec",
            "meets_mse_le_1",
            "meets_biozero_le_thresh",
            "meets_joint_cells100",
        ],
        rows,
    )
    return rows


def merge_with_scope5(cells100_rows: List[Dict[str, object]]) -> None:
    scope_rows = _read_tsv(SCOPE5_OUT / "sweep_results.tsv")
    c_map = {str(r["run_name"]): r for r in cells100_rows}

    merged_rows: List[Dict[str, object]] = []
    for s in scope_rows:
        run_name = str(s.get("run_name", ""))
        c = c_map.get(run_name, {})
        merged_rows.append(
            {
                "run_name": run_name,
                "option": int(_safe_float(s.get("option"))),
                "avg_ari": _safe_float(s.get("avg_ari")),
                "wins_vs_benchmark_best": int(_safe_float(s.get("wins_vs_benchmark_best"))),
                "cells100_avg_mse": _safe_float(c.get("cells100_avg_mse")),
                "cells100_avg_biozero": _safe_float(c.get("cells100_avg_biozero")),
                "meets_mse_le_1": int(_safe_float(c.get("meets_mse_le_1"))),
                "meets_biozero_le_thresh": int(_safe_float(c.get("meets_biozero_le_thresh"))),
                "meets_joint_cells100": int(_safe_float(c.get("meets_joint_cells100"))),
            }
        )

    _write_tsv(
        SCOPE5_OUT / "sweep_results_with_cells100.tsv",
        [
            "run_name",
            "option",
            "avg_ari",
            "wins_vs_benchmark_best",
            "cells100_avg_mse",
            "cells100_avg_biozero",
            "meets_mse_le_1",
            "meets_biozero_le_thresh",
            "meets_joint_cells100",
        ],
        merged_rows,
    )

    best_rows: List[Dict[str, object]] = []
    for option in sorted({int(r["option"]) for r in merged_rows}):
        subset = [r for r in merged_rows if int(r["option"]) == option]
        subset.sort(
            key=lambda r: (
                int(r["meets_joint_cells100"]),
                float(r["avg_ari"]),
                -float(r["cells100_avg_biozero"]),
                -float(r["cells100_avg_mse"]),
            ),
            reverse=True,
        )
        best = subset[0]
        best_rows.append(best)

    _write_tsv(
        SCOPE5_OUT / "best_by_option_with_cells100.tsv",
        [
            "run_name",
            "option",
            "avg_ari",
            "wins_vs_benchmark_best",
            "cells100_avg_mse",
            "cells100_avg_biozero",
            "meets_mse_le_1",
            "meets_biozero_le_thresh",
            "meets_joint_cells100",
        ],
        best_rows,
    )


def main() -> None:
    cells100_rows = run_cells100()
    merge_with_scope5(cells100_rows)
    print("\n[DONE] Wrote:")
    print(" -", CELLS100_OUT / "cells100_metrics.tsv")
    print(" -", SCOPE5_OUT / "sweep_results_with_cells100.tsv")
    print(" -", SCOPE5_OUT / "best_by_option_with_cells100.tsv")
    print(" - biozero threshold:", BIOZERO_THRESH)


if __name__ == "__main__":
    main()
