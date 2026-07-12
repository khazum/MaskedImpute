#!/usr/bin/env python3
"""
run_imputation.py
-----------------

Runs multiple imputation methods (MAGIC, legacy DCA, scVI, ALRA, AutoClass, MaskImpute)
on SingleCellExperiment .rds files and reports reconstruction metrics in
log2(1+normalized) space vs. logTrueCounts.

Reported metrics follow benchmark_datasets_and_metrics.md:
- MSE (overall, dropout, biozero, non-zero)
- MSE on marker genes (DEFacGroup columns in rowData)
- MAE (overall, dropout, biozero, non-zero)
- MAE on marker genes (DEFacGroup columns in rowData)
- per-gene normalized RMSE (gNRMSE)
- gNRMSE on marker genes (DEFacGroup columns in rowData)
- gene-gene correlation error (CorrErr)

Count outputs are normalized with dataset `target_sum` and library sizes
recomputed from each imputed count matrix (no libSizeTrue usage).

Required Python packages: rds2py, numpy.
Method-specific dependencies: magic-impute (MAGIC), pandas + DCA CLI (DCA bridge),
scvi-tools + anndata (scVI), scikit-learn (ALRA), and AutoClass with its
Python requirements (e.g., tensorflow).
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import os
import subprocess
import time
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

warnings.filterwarnings(
    "ignore",
    message=r"Unsupported R object type: other",
    category=RuntimeWarning,
    module=r"rds2py\.PyRdsReader",
)

try:
    from rds2py import read_rds
except Exception as exc:  # pragma: no cover - import error surfaced for user
    raise SystemExit(
        "Failed to import rds2py in this Python. See script header for requirements.\n"
        f"Python: {sys.executable}\n"
        f"Error: {exc}\n"
    ) from exc

try:
    from rds2py.PyRdsReader import PyRdsParser
    from rds2py.generics import _dispatcher as _rds_dispatcher
except Exception:
    PyRdsParser = None
    _rds_dispatcher = None

EPSILON = 1e-8
METHODS = ("magic", "dca", "scvi", "alra", "autoclass", "balanced_mse")
METHOD_DISPLAY_NAMES = {
    "magic": "MAGIC",
    "dca": "DCA",
    "scvi": "scVI",
    "alra": "ALRA",
    "autoclass": "AutoClass",
    # Historical result key; public approach name is MaskImpute.
    "balanced_mse": "MaskImpute",
}
METHOD_ALIASES = {
    "maskimpute": "balanced_mse",
    "maskedimpute": "balanced_mse",
    "masked_impute": "balanced_mse",
    "masked_imputation26": "balanced_mse",
}
METHOD_HELP = "magic,dca,scvi,alra,autoclass,balanced_mse (MaskImpute),maskimpute"


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    n = int(mask.sum())
    if n == 0:
        return float("nan")
    return float(np.mean(values[mask]))


def compute_masks(
    log_true: np.ndarray,
    log_obs: np.ndarray,
    marker_gene_mask: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    n_genes = int(log_true.shape[1])
    if marker_gene_mask is None:
        marker_gene = np.zeros(n_genes, dtype=bool)
    else:
        marker_gene = np.asarray(marker_gene_mask, dtype=bool).reshape(-1)
        if marker_gene.size != n_genes:
            raise ValueError(
                f"marker gene mask length {marker_gene.size} does not match number of genes {n_genes}."
            )
    marker_mask = np.broadcast_to(marker_gene[None, :], log_true.shape)
    return {
        "biozero": log_true <= EPSILON,
        "dropout": (log_true > EPSILON) & (log_obs <= EPSILON),
        "non_zero": (log_true > EPSILON) & (log_obs > EPSILON),
        "marker": marker_mask,
        "marker_gene": marker_gene,
    }


def compute_gnrmse(
    log_imp: np.ndarray,
    log_true: np.ndarray,
    gene_mask: Optional[np.ndarray] = None,
) -> float:
    # Matrices are cells x genes. gNRMSE is aggregated across genes.
    diff = np.asarray(log_true - log_imp, dtype=np.float64)
    rmse_gene = np.sqrt(np.mean(diff ** 2, axis=0))
    sd_true = np.std(np.asarray(log_true, dtype=np.float64), axis=0, ddof=0)
    denom = np.maximum(sd_true, EPSILON)
    vals = rmse_gene / denom
    if gene_mask is not None:
        gm = np.asarray(gene_mask, dtype=bool).reshape(-1)
        if gm.size != vals.size:
            raise ValueError(f"gNRMSE gene mask length {gm.size} does not match genes {vals.size}.")
        vals = vals[gm]
    finite = np.isfinite(vals)
    if not np.any(finite):
        return float("nan")
    return float(np.mean(vals[finite]))


def compute_corr_err(log_imp: np.ndarray, log_true: np.ndarray) -> Tuple[float, int]:
    # Pearson gene-gene correlation across cells, filtering zero-variance genes.
    true = np.asarray(log_true, dtype=np.float64)
    imp = np.asarray(log_imp, dtype=np.float64)

    sd_true = np.std(true, axis=0, ddof=0)
    sd_imp = np.std(imp, axis=0, ddof=0)
    keep = np.isfinite(sd_true) & np.isfinite(sd_imp) & (sd_true > EPSILON) & (sd_imp > EPSILON)
    g_corr = int(keep.sum())
    if g_corr < 2:
        return float("nan"), g_corr

    true_sub = true[:, keep]
    imp_sub = imp[:, keep]
    r_true = np.corrcoef(true_sub, rowvar=False)
    r_imp = np.corrcoef(imp_sub, rowvar=False)
    tri = np.triu_indices(g_corr, k=1)
    diff = np.abs(r_true[tri] - r_imp[tri])
    if diff.size == 0 or not np.all(np.isfinite(diff)):
        return float("nan"), g_corr
    return float(np.mean(diff)), g_corr


def compute_error_metrics(
    log_imp: np.ndarray, log_true: np.ndarray, masks: Dict[str, np.ndarray]
) -> Dict[str, float]:
    if log_imp.shape != log_true.shape:
        raise ValueError(f"log_imp shape {log_imp.shape} does not match log_true {log_true.shape}.")
    if masks["biozero"].shape != log_true.shape:
        raise ValueError("Mask shape does not match log_true.")
    if masks["marker"].shape != log_true.shape:
        raise ValueError("marker mask shape does not match log_true.")
    if np.asarray(masks["marker_gene"]).reshape(-1).size != log_true.shape[1]:
        raise ValueError("marker gene mask length does not match number of genes.")
    diff = np.asarray(log_true - log_imp, dtype=np.float64)
    sq_diff = diff ** 2
    abs_diff = np.abs(diff)
    corr_err, n_corr_genes = compute_corr_err(log_imp, log_true)

    return {
        "mse": float(np.mean(sq_diff)),
        "mse_dropout": _masked_mean(sq_diff, masks["dropout"]),
        "mse_biozero": _masked_mean(sq_diff, masks["biozero"]),
        "mse_non_zero": _masked_mean(sq_diff, masks["non_zero"]),
        "mse_marker": _masked_mean(sq_diff, masks["marker"]),
        "mae": float(np.mean(abs_diff)),
        "mae_dropout": _masked_mean(abs_diff, masks["dropout"]),
        "mae_biozero": _masked_mean(abs_diff, masks["biozero"]),
        "mae_non_zero": _masked_mean(abs_diff, masks["non_zero"]),
        "mae_marker": _masked_mean(abs_diff, masks["marker"]),
        "gnrmse": compute_gnrmse(log_imp, log_true),
        "gnrmse_marker": compute_gnrmse(log_imp, log_true, masks["marker_gene"]),
        "corr_err": corr_err,
        "n_corr_genes": n_corr_genes,
        "n_total": int(diff.size),
        "n_dropout": int(masks["dropout"].sum()),
        "n_biozero": int(masks["biozero"].sum()),
        "n_non_zero": int(masks["non_zero"].sum()),
        "n_marker": int(masks["marker"].sum()),
        "n_marker_genes": int(np.asarray(masks["marker_gene"], dtype=bool).sum()),
    }


def compute_mask_counts(masks: Dict[str, np.ndarray]) -> Dict[str, int]:
    return {
        "n_total": int(masks["biozero"].size),
        "n_dropout": int(masks["dropout"].sum()),
        "n_biozero": int(masks["biozero"].sum()),
        "n_non_zero": int(masks["non_zero"].sum()),
        "n_marker": int(masks["marker"].sum()),
        "n_marker_genes": int(np.asarray(masks["marker_gene"], dtype=bool).sum()),
    }


def _to_dense_array(x: object) -> np.ndarray:
    if hasattr(x, "toarray"):
        x = x.toarray()
    return np.asarray(x)


def normalize_counts_to_logcounts(
    counts_mat: np.ndarray,
    target_sum: float,
) -> np.ndarray:
    """Normalize counts to log2(1+CPtarget) using imputed per-cell library sizes."""
    x = np.asarray(counts_mat, dtype=np.float64)
    x = np.clip(x, 0.0, None)
    ts = float(target_sum)
    if not np.isfinite(ts) or ts <= 0.0:
        raise ValueError(f"Invalid target_sum for normalization: {target_sum}")
    lib_sizes = np.sum(x, axis=1)
    denom = np.where(lib_sizes > 0.0, lib_sizes / ts, 1.0)
    norm = x / denom[:, None]
    return np.log2(1.0 + norm).astype(np.float32)


def _get_metadata_normalization(sce) -> Dict[str, object]:
    md = getattr(sce, "metadata", None)
    if isinstance(md, dict):
        norm = md.get("normalization")
        if isinstance(norm, dict):
            return norm
    return {}


def _get_coldata_column(sce, name: str) -> Optional[np.ndarray]:
    coldata = getattr(sce, "column_data", None)
    if coldata is None:
        return None
    try:
        if hasattr(coldata, "column"):
            return np.asarray(coldata.column(name))
        return np.asarray(coldata[name])
    except Exception:
        return None


def _extract_marker_gene_mask_from_sce_rowdata(sce, n_genes: int) -> Optional[np.ndarray]:
    row_data = getattr(sce, "row_data", None)
    if row_data is None:
        return None
    try:
        col_names = [str(x) for x in getattr(row_data, "column_names", [])]
    except Exception:
        return None
    defac_cols = [cn for cn in col_names if cn.startswith("DEFac")]
    if not defac_cols:
        return None

    mask = np.zeros(int(n_genes), dtype=bool)
    for cn in defac_cols:
        try:
            vals = np.asarray(row_data.column(cn), dtype=np.float64).reshape(-1)
        except Exception:
            continue
        if vals.size != mask.size:
            continue
        mask |= np.isfinite(vals) & (np.abs(vals - 1.0) > EPSILON)
    return mask


def _extract_marker_gene_mask_from_rds(path: str, n_genes: int) -> Optional[np.ndarray]:
    if PyRdsParser is None or _rds_dispatcher is None:
        return None

    try:
        raw = PyRdsParser(path).parse()
    except Exception:
        return None

    attrs = raw.get("attributes", {}) if isinstance(raw, dict) else {}
    candidates: List[object] = []
    row_ranges = attrs.get("rowRanges")
    if isinstance(row_ranges, dict):
        rr_attrs = row_ranges.get("attributes", {})
        if isinstance(rr_attrs, dict):
            candidates.append(rr_attrs.get("elementMetadata"))
    candidates.append(attrs.get("elementMetadata"))

    for node in candidates:
        if not isinstance(node, dict):
            continue
        try:
            frame = _rds_dispatcher(node)
            col_names = [str(x) for x in getattr(frame, "column_names", [])]
        except Exception:
            continue
        defac_cols = [cn for cn in col_names if cn.startswith("DEFac")]
        if not defac_cols:
            continue

        n = int(n_genes)
        marker = np.zeros(n, dtype=bool)
        for cn in defac_cols:
            try:
                vals = np.asarray(frame.column(cn), dtype=np.float64).reshape(-1)
            except Exception:
                continue
            if vals.size != n:
                continue
            marker |= np.isfinite(vals) & (np.abs(vals - 1.0) > EPSILON)
        return marker

    return None


def _infer_target_sum_from_sce(sce) -> Optional[float]:
    try:
        counts = _to_dense_array(sce.assay("counts")).astype(np.float64, copy=False)  # genes x cells
        log_obs = _to_dense_array(sce.assay("logcounts")).astype(np.float64, copy=False)  # genes x cells
    except Exception:
        return None

    if counts.shape != log_obs.shape or counts.ndim != 2:
        return None
    n_genes, n_cells = counts.shape
    if n_genes == 0 or n_cells == 0:
        return None

    # Sample a grid to keep inference cheap on very large matrices.
    max_genes = 256
    max_cells = 2048
    g_step = max(1, n_genes // max_genes)
    c_step = max(1, n_cells // max_cells)
    g_idx = np.arange(0, n_genes, g_step, dtype=int)[:max_genes]
    c_idx = np.arange(0, n_cells, c_step, dtype=int)[:max_cells]

    sub_counts = counts[np.ix_(g_idx, c_idx)]
    sub_log = log_obs[np.ix_(g_idx, c_idx)]
    lib_sizes = np.sum(counts, axis=0)[c_idx]

    mask = (
        (sub_counts > 0.0)
        & np.isfinite(sub_log)
        & np.isfinite(lib_sizes)[None, :]
        & (lib_sizes[None, :] > 0.0)
    )
    if not np.any(mask):
        return None

    numer = np.expm1(sub_log[mask] * np.log(2.0))
    denom_counts = sub_counts[mask]
    cell_ids = np.where(mask)[1]
    vals = numer * lib_sizes[cell_ids] / denom_counts
    vals = vals[np.isfinite(vals) & (vals > 0.0)]
    if vals.size == 0:
        return None
    return float(np.median(vals))


def get_normalization_info(sce) -> Dict[str, float]:
    """Return normalization constants used for benchmark evaluation."""
    md_norm = _get_metadata_normalization(sce)
    target_sum = None

    # Prefer explicit colData scalar replicated per cell when available.
    target_col = _get_coldata_column(sce, "targetSum")
    if target_col is not None:
        vals = np.asarray(target_col, dtype=np.float64)
        vals = vals[np.isfinite(vals) & (vals > 0.0)]
        if vals.size:
            target_sum = float(vals[0])

    if target_sum is None:
        target_sum = md_norm.get("target_sum")

    if target_sum is None:
        inferred = _infer_target_sum_from_sce(sce)
        if inferred is not None:
            target_sum = inferred
            print(f"[WARN] normalization.target_sum missing; inferred target_sum={target_sum:.6g} from counts/logcounts.")

    if target_sum is None:
        raise RuntimeError("Missing normalization.target_sum metadata and could not infer it from counts/logcounts.")

    target_sum = float(target_sum)
    if not np.isfinite(target_sum) or target_sum <= 0.0:
        raise RuntimeError(f"Invalid normalization.target_sum: {target_sum}")
    return {"target_sum": target_sum}


def load_dataset(path: str) -> Optional[Dict[str, object]]:
    sce = read_rds(path)
    if not hasattr(sce, "assay"):
        raise TypeError(f"Unsupported RDS object (expected SingleCellExperiment): {type(sce)}")

    logcounts = _to_dense_array(sce.assay("logcounts")).T.astype("float32", copy=False)

    log_true = None
    for assay_name in ("logTrueCounts", "perfect_logcounts"):
        try:
            log_true = _to_dense_array(sce.assay(assay_name)).T.astype("float32", copy=False)
            break
        except Exception:
            continue

    if log_true is None:
        return None

    counts = None
    try:
        counts = _to_dense_array(sce.assay("counts")).T.astype("float32", copy=False)
    except Exception:
        counts = None

    norm_info = get_normalization_info(sce)

    marker_gene_mask = _extract_marker_gene_mask_from_sce_rowdata(sce, logcounts.shape[1])
    if marker_gene_mask is None:
        marker_gene_mask = _extract_marker_gene_mask_from_rds(path, logcounts.shape[1])
    if marker_gene_mask is None:
        marker_gene_mask = np.zeros(logcounts.shape[1], dtype=bool)
        print("[WARN] DEFacGroup rowData missing/unreadable; marker-subset metrics will be NA.")

    if logcounts.shape != log_true.shape:
        raise ValueError(f"Assay shape mismatch: logcounts {logcounts.shape} vs logTrueCounts {log_true.shape}.")
    if counts is not None and counts.shape != logcounts.shape:
        raise ValueError(f"Assay shape mismatch: counts {counts.shape} vs logcounts {logcounts.shape}.")

    return {
        "logcounts": logcounts,
        "log_true": log_true,
        "counts": counts,
        "norm": norm_info,
        "marker_gene_mask": marker_gene_mask,
    }


def collect_rds_files(input_path: str) -> List[Path]:
    path = Path(input_path)
    if path.is_dir():
        return sorted(path.rglob("*.rds"))
    if path.is_file():
        return [path]
    raise FileNotFoundError(f"Input path not found: {input_path}")


def dataset_name_from_path(path: Path) -> str:
    stem = path.stem
    if stem.lower() != "sce":
        return stem

    parent = path.parent.name
    grandparent = path.parent.parent.name
    if grandparent in {"test", "tune"} and parent:
        return parent
    if parent.startswith("n") and grandparent:
        return f"{grandparent}_{parent}"
    return parent or stem


# ---------------------------- MAGIC ----------------------------

def _import_magic():
    try:
        import magic
    except Exception as exc:  # pragma: no cover - import error surfaced for user
        raise SystemExit(
            "Failed to import magic-impute in this Python. See script header for requirements.\n"
            f"Python: {sys.executable}\n"
            f"Error: {exc}\n"
        ) from exc

    if not hasattr(magic, "MAGIC"):  # pragma: no cover - defensive check
        magic_path = getattr(magic, "__file__", "unknown")
        raise SystemExit(
            "Imported a module named 'magic' but it does not provide MAGIC.\n"
            f"Loaded from: {magic_path}\n"
            "This usually means python-magic (libmagic) is shadowing magic-impute."
        )

    return magic


def run_magic(logcounts: np.ndarray, n_jobs: int) -> np.ndarray:
    magic = _import_magic()
    try:
        op = magic.MAGIC(n_jobs=int(n_jobs))
    except Exception:
        op = magic.MAGIC()
    out = op.fit_transform(logcounts)
    return np.asarray(out, dtype=np.float32)


# ---------------------------- DCA (Legacy Bridge) ----------------------------

def _import_pandas():
    try:
        import pandas as pd
    except Exception as exc:  # pragma: no cover - import error surfaced for user
        raise SystemExit(
            "Failed to import pandas (required for DCA bridge). See script header for requirements.\n"
            f"Python: {sys.executable}\n"
            f"Error: {exc}\n"
        ) from exc
    return pd


def _read_table(path: Path) -> np.ndarray:
    pd = _import_pandas()
    attempts = [
        {"sep": "\t", "index_col": 0},
        {"sep": ",", "index_col": 0},
        {"sep": None, "engine": "python", "index_col": 0},
        {"sep": "\t", "index_col": None},
        {"sep": ",", "index_col": None},
        {"sep": None, "engine": "python", "index_col": None},
    ]
    for kw_read in attempts:
        try:
            df_try = pd.read_csv(path, **kw_read)
            if df_try.shape[0] > 0 and df_try.shape[1] > 0:
                return df_try.values.astype(np.float64)
        except Exception:
            continue
    raise RuntimeError(f"Could not parse {path}")


def run_dca(
    counts: np.ndarray,
    dca_bin: str,
    ae_type: Optional[str],
    epochs: Optional[int],
    batch_size: Optional[int],
    threads: Optional[int],
    ridge: Optional[float],
    verbose: bool,
    timeout_sec: Optional[float] = None,
) -> np.ndarray:
    if dca_bin is None:
        raise RuntimeError("DCA binary not provided.")
    dca_bin = str(Path(dca_bin).expanduser())
    if not os.path.exists(dca_bin):
        raise RuntimeError(f"DCA binary not found at: {dca_bin}")
    cuda_env = os.environ.get("CUDA_VISIBLE_DEVICES") or "<unset>"
    print(f"[DCA] CUDA_VISIBLE_DEVICES={cuda_env} dca_bin={dca_bin}")

    pd = _import_pandas()
    C = np.rint(np.asarray(counts)).astype(np.int32, copy=False)
    cell_mask = (C.sum(axis=1) > 0)
    gene_mask = (C.sum(axis=0) > 0)
    n_drop_cells = int((~cell_mask).sum())
    n_drop_genes = int((~gene_mask).sum())
    C_work = C[np.ix_(cell_mask, gene_mask)] if (n_drop_cells or n_drop_genes) else C

    if n_drop_cells or n_drop_genes:
        if verbose:
            print(f"   [DCA] Removing {n_drop_cells} zero-count cells and {n_drop_genes} zero-count genes before DCA.")

    if C_work.shape[0] < 2 or C_work.shape[1] < 2:
        raise ValueError(f"DCA requires at least 2 cells and 2 genes. Got shape {C_work.shape}.")

    def _dca_python_bin(dca_exec: str) -> Optional[str]:
        bin_dir = Path(dca_exec).resolve().parent
        py = bin_dir / "python"
        if py.exists():
            return str(py)
        return None

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_dir_path = Path(tmpdir)
        input_file = tmp_dir_path / "matrix.csv"
        output_dir = tmp_dir_path / "dca_out"
        wrapper_file = tmp_dir_path / "dca_yaml_compat_wrapper.py"

        try:
            gene_names = [f"gene_{i}" for i in range(C_work.shape[1])]
            cell_names = [f"cell_{i}" for i in range(C_work.shape[0])]
            df = pd.DataFrame(C_work.T, index=gene_names, columns=cell_names)
            df.to_csv(input_file, sep=",")
        except Exception as e:
            raise RuntimeError(f"Failed to write input CSV for DCA: {e}")

        cmd_args = [str(input_file), str(output_dir)]

        if ae_type is not None:
            cmd_args.extend(["--type", ae_type])
        if threads is not None:
            cmd_args.extend(["--threads", str(threads)])
        if epochs is not None:
            cmd_args.extend(["-e", str(epochs)])
        if batch_size is not None:
            cmd_args.extend(["-b", str(batch_size)])
        if ridge is not None:
            cmd_args.extend(["--ridge", str(ridge)])

        py_bin = _dca_python_bin(dca_bin)
        if py_bin:
            wrapper_file.write_text(
                "import sys\n"
                "import yaml\n"
                "_orig_load = yaml.load\n"
                "def _compat_load(stream, Loader=None, *args, **kwargs):\n"
                "    if Loader is None:\n"
                "        return yaml.safe_load(stream)\n"
                "    return _orig_load(stream, Loader=Loader, *args, **kwargs)\n"
                "yaml.load = _compat_load\n"
                "from dca.__main__ import main\n"
                "sys.argv = ['dca'] + sys.argv[1:]\n"
                "sys.exit(main())\n",
                encoding="utf-8",
            )
            cmd = [py_bin, str(wrapper_file)] + cmd_args
        else:
            cmd = [dca_bin] + cmd_args

        try:
            subprocess.run(
                cmd,
                check=True,
                stdout=None if verbose else subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired as e:
            raise TimeoutError(f"DCA timed out after {timeout_sec} seconds.") from e
        except subprocess.CalledProcessError as e:
            err_msg = e.stderr.decode("utf-8") if e.stderr else ""
            mean_exists = (output_dir / "mean.tsv").exists()
            if mean_exists and ("Shape of passed values" in err_msg or "ValueError" in err_msg):
                if verbose:
                    print("   [DCA Warning] DCA crashed during writing (known legacy bug). Recovering...")
            else:
                raise RuntimeError(f"DCA Bridge failed:\n{err_msg}")

        try:
            mu = _read_table(output_dir / "mean.tsv").T
        except Exception as e:
            raise RuntimeError(f"Failed to read DCA output: {e}")

    def _coerce_shape(arr: np.ndarray, target: Tuple[int, int], fill: float = 0.0) -> np.ndarray:
        if arr.shape == target:
            return arr
        if arr.T.shape == target:
            return arr.T
        work = arr
        if work.shape[1] + 1 == target[1]:
            padded = np.full((work.shape[0], target[1]), fill, dtype=work.dtype)
            padded[:, :-1] = work
            work = padded
        if work.shape[0] == 1 and target[0] > 1:
            try:
                work = np.broadcast_to(work, (target[0], work.shape[1]))
            except Exception:
                pass
        try:
            return np.broadcast_to(work, target)
        except Exception:
            raise ValueError(f"Shape mismatch: {arr.shape} vs {target}")

    mu = _coerce_shape(mu, C_work.shape, 0.0)

    if n_drop_cells or n_drop_genes:
        full_shape = C.shape
        mu_full = np.zeros(full_shape, dtype=np.float32)
        idx = np.ix_(cell_mask, gene_mask)
        mu_full[idx] = mu.astype(np.float32)
        return mu_full

    return mu.astype(np.float32)


# ---------------------------- scVI ----------------------------

def _import_scvi():
    try:
        import anndata as ad
        import scvi
    except Exception as exc:  # pragma: no cover - import error surfaced for user
        raise SystemExit(
            "Failed to import scvi-tools/anndata for scVI baseline.\n"
            "Use a Python environment with scvi-tools installed, or set SCVI_PYTHON in the parallel runner.\n"
            f"Python: {sys.executable}\n"
            f"Error: {exc}\n"
        ) from exc
    return scvi, ad


def _call_train_scvi(model, **kwargs) -> None:
    """Call scVI train while tolerating minor API differences across versions."""
    train_kwargs = dict(kwargs)
    for _ in range(8):
        try:
            model.train(**train_kwargs)
            return
        except TypeError as exc:
            msg = str(exc)
            removed = False
            for key in (
                "enable_progress_bar",
                "enable_checkpointing",
                "logger",
                "default_root_dir",
                "accelerator",
                "devices",
                "batch_size",
            ):
                if key in train_kwargs and key in msg:
                    train_kwargs.pop(key, None)
                    removed = True
                    break
            if not removed:
                raise


def run_scvi(
    counts: np.ndarray,
    target_sum: float,
    *,
    seed: int,
    max_epochs: int,
    n_latent: int,
    n_hidden: int,
    n_layers: int,
    batch_size: int,
) -> np.ndarray:
    scvi, ad = _import_scvi()
    try:
        import torch
    except Exception:
        torch = None

    C = np.rint(np.asarray(counts, dtype=np.float32))
    C = np.clip(C, 0.0, None)
    cell_mask = np.sum(C, axis=1) > 0.0
    gene_mask = np.sum(C, axis=0) > 0.0
    if int(cell_mask.sum()) < 2 or int(gene_mask.sum()) < 2:
        raise ValueError(f"scVI requires at least 2 nonzero cells and genes. Got {C.shape}.")

    C_work = C[np.ix_(cell_mask, gene_mask)]
    try:
        scvi.settings.seed = int(seed)
    except Exception:
        pass
    if torch is not None:
        try:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))
        except Exception:
            pass

    adata = ad.AnnData(X=C_work.astype(np.float32, copy=False))
    adata.layers["counts"] = C_work.astype(np.float32, copy=False)
    scvi.model.SCVI.setup_anndata(adata, layer="counts")
    model = scvi.model.SCVI(
        adata,
        n_layers=int(n_layers),
        n_hidden=int(n_hidden),
        n_latent=int(n_latent),
        gene_likelihood="nb",
    )

    accelerator = "cpu"
    if torch is not None:
        try:
            accelerator = "gpu" if torch.cuda.is_available() else "cpu"
        except Exception:
            accelerator = "cpu"

    with tempfile.TemporaryDirectory() as tmpdir:
        _call_train_scvi(
            model,
            max_epochs=int(max_epochs),
            batch_size=int(batch_size),
            accelerator=accelerator,
            devices=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            default_root_dir=tmpdir,
        )

    expr = None
    attempts = (
        {"library_size": float(target_sum), "return_numpy": True},
        {"library_size": float(target_sum)},
        {"return_numpy": True},
        {},
    )
    last_exc: Optional[Exception] = None
    for kwargs in attempts:
        try:
            expr = model.get_normalized_expression(**kwargs)
            break
        except Exception as exc:
            last_exc = exc
    if expr is None:
        raise RuntimeError(f"scVI failed to produce normalized expression: {last_exc}")

    if hasattr(expr, "values"):
        expr = expr.values
    expr_arr = np.asarray(expr, dtype=np.float32)
    if expr_arr.shape != C_work.shape and expr_arr.T.shape == C_work.shape:
        expr_arr = expr_arr.T
    if expr_arr.shape != C_work.shape:
        raise ValueError(f"scVI output shape {expr_arr.shape} does not match working counts {C_work.shape}.")

    # If the API returned unit-scale normalized expression, rescale to the benchmark target.
    row_sums = np.sum(expr_arr, axis=1)
    median_sum = float(np.nanmedian(row_sums[row_sums > 0])) if np.any(row_sums > 0) else 0.0
    if np.isfinite(median_sum) and median_sum > 0.0 and median_sum < 0.1 * float(target_sum):
        expr_arr = expr_arr * float(target_sum)

    full_norm = np.zeros_like(C, dtype=np.float32)
    full_norm[np.ix_(cell_mask, gene_mask)] = np.clip(expr_arr, 0.0, None)
    return np.log2(1.0 + full_norm).astype(np.float32)


# ---------------------------- ALRA ----------------------------

def _alra_rank_from_spectrum(svals: np.ndarray, n_cells: int, n_genes: int, max_rank: int) -> int:
    finite = np.asarray(svals, dtype=np.float64)
    finite = finite[np.isfinite(finite) & (finite > 0.0)]
    if finite.size == 0:
        return 1
    beta = min(n_cells, n_genes) / max(n_cells, n_genes)
    # Gavish-Donoho hard threshold coefficient for unknown noise level.
    omega = 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43
    threshold = omega * float(np.median(finite))
    rank = int(np.sum(finite > threshold))
    if rank < 1:
        # Fall back to a conservative elbow-like rank when the spectrum is flat.
        ratios = finite[:-1] / np.maximum(finite[1:], EPSILON)
        rank = int(np.argmax(ratios) + 1) if ratios.size else 1
    return int(np.clip(rank, 1, max(1, min(max_rank, finite.size))))


def run_alra(
    logcounts: np.ndarray,
    *,
    seed: int,
    rank: Optional[int],
    max_rank: int,
    n_iter: int,
) -> np.ndarray:
    from sklearn.utils.extmath import randomized_svd

    X = np.asarray(logcounts, dtype=np.float32)
    if X.ndim != 2 or min(X.shape) < 2:
        raise ValueError(f"ALRA requires a 2D matrix with at least 2 cells/genes. Got {X.shape}.")

    max_components = int(max(1, min(max_rank, min(X.shape) - 1)))
    gene_means = np.mean(X, axis=0, keepdims=True)
    X_centered = X - gene_means
    U, svals, Vt = randomized_svd(
        X_centered,
        n_components=max_components,
        n_iter=int(n_iter),
        random_state=int(seed),
    )
    k = int(rank) if rank is not None and int(rank) > 0 else _alra_rank_from_spectrum(svals, X.shape[0], X.shape[1], max_components)
    k = int(np.clip(k, 1, max_components))
    recon = (U[:, :k] * svals[:k]) @ Vt[:k, :]
    recon = recon + gene_means

    # Adaptive thresholding: use each gene's negative reconstruction tail to
    # remove low positive values likely introduced by low-rank smoothing.
    out = recon.astype(np.float32, copy=False)
    thresholds = np.zeros(out.shape[1], dtype=np.float32)
    for j in range(out.shape[1]):
        neg = out[:, j][out[:, j] < 0.0]
        if neg.size:
            thresholds[j] = float(np.quantile(-neg, 0.999))
    out = np.where(out <= thresholds[None, :], 0.0, out)
    out = np.clip(out, 0.0, None)
    return out.astype(np.float32)


# ---------------------------- AutoClass ----------------------------

def _import_autoclass(autoclass_dir: Optional[str]):
    if not autoclass_dir:
        raise RuntimeError("AutoClass directory not provided.")

    path = Path(autoclass_dir).expanduser().resolve()
    if not path.exists():
        raise RuntimeError(f"AutoClass directory not found: {path}")

    sys.path.insert(0, str(path))
    src_path = path / "AutoClass_src"
    if src_path.exists():
        sys.path.insert(0, str(src_path))

    tried = []
    for mod_name in ("AutoClass_src.AutoClass.AutoClass", "AutoClass.AutoClass", "AutoClass"):
        try:
            module = importlib.import_module(mod_name)
            return module
        except Exception as exc:
            tried.append(f"{mod_name}: {exc}")

    msg = "\n".join(tried)
    raise RuntimeError(
        "Failed to import AutoClass. Ensure the repo is cloned and on PYTHONPATH.\n"
        f"Tried:\n{msg}"
    )


def _resolve_autoclass_callable(module):
    if hasattr(module, "AutoClassImpute"):
        return getattr(module, "AutoClassImpute")
    if hasattr(module, "AutoClass"):
        return getattr(module, "AutoClass")
    for name in ("run_AutoClass", "run_autoclass", "autoclass"):
        if hasattr(module, name):
            return getattr(module, name)
    raise RuntimeError("AutoClass import succeeded but no callable entrypoint was found.")


def _parse_kv_pairs(raw: str) -> Dict[str, object]:
    if not raw:
        return {}
    out: Dict[str, object] = {}
    for part in raw.split(","):
        if not part.strip():
            continue
        if "=" not in part:
            raise ValueError(f"Invalid --autoclass-kwargs entry: '{part}'. Expected key=value.")
        key, value = part.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(f"Invalid --autoclass-kwargs entry: '{part}'.")
        lowered = value.lower()
        if lowered in {"true", "false"}:
            out[key] = lowered == "true"
        else:
            try:
                out[key] = int(value)
            except ValueError:
                try:
                    out[key] = float(value)
                except ValueError:
                    out[key] = value
    return out


def _extract_autoclass_output(obj: object, data: np.ndarray) -> Optional[np.ndarray]:
    if isinstance(obj, dict):
        for key in ("imp", "imputed", "denoised", "output", "result"):
            if key in obj:
                arr = np.asarray(obj[key])
                if arr.shape == data.shape or arr.T.shape == data.shape:
                    return arr.T if arr.shape != data.shape else arr
    for attr in ("imputed", "denoised", "output", "result", "X_imputed", "X_recon"):
        if hasattr(obj, attr):
            arr = np.asarray(getattr(obj, attr))
            if arr.shape == data.shape or arr.T.shape == data.shape:
                return arr.T if arr.shape != data.shape else arr
    return None


def run_autoclass(logcounts: np.ndarray, autoclass_dir: Optional[str], autoclass_kwargs: Dict[str, object]) -> np.ndarray:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    module = _import_autoclass(autoclass_dir)
    entry = _resolve_autoclass_callable(module)

    run_kwargs = dict(autoclass_kwargs)
    run_kwargs.setdefault("log1p", False)
    run_kwargs.setdefault("cellwise_norm", False)
    run_kwargs.setdefault("verbose", False)
    if "num_cluster" not in run_kwargs:
        run_kwargs["num_cluster"] = [8, 9, 10]
    else:
        num_cluster = run_kwargs["num_cluster"]
        if isinstance(num_cluster, int):
            n = int(num_cluster)
            run_kwargs["num_cluster"] = [max(1, n - 1), n, n + 1]
        elif isinstance(num_cluster, (list, tuple)):
            run_kwargs["num_cluster"] = [int(v) for v in num_cluster]

    if inspect.isclass(entry):
        try:
            model = entry(logcounts.copy(), **run_kwargs)
        except TypeError:
            model = entry(**run_kwargs)
        if hasattr(model, "fit_transform"):
            out = model.fit_transform(logcounts)
        elif hasattr(model, "fit") and hasattr(model, "transform"):
            model.fit(logcounts)
            out = model.transform(logcounts)
        elif hasattr(model, "fit") and hasattr(model, "predict"):
            model.fit(logcounts)
            out = model.predict(logcounts)
        else:
            extracted = _extract_autoclass_output(model, logcounts)
            if extracted is None:
                raise RuntimeError("AutoClass instance has no recognized fit/transform/predict method.")
            out = extracted
    else:
        try:
            out = entry(logcounts.copy(), **run_kwargs)
        except TypeError:
            out = entry(logcounts.copy())

    extracted = _extract_autoclass_output(out, logcounts)
    if extracted is not None:
        out = extracted

    out = np.asarray(out)
    if out.shape != logcounts.shape and out.T.shape == logcounts.shape:
        out = out.T
    if out.shape != logcounts.shape:
        raise ValueError(f"AutoClass output shape {out.shape} does not match input {logcounts.shape}.")
    return out.astype(np.float32)


def _import_maskimpute_impl():
    try:
        import masked_imputation26 as mi26
    except Exception as exc:  # pragma: no cover - import error surfaced for user
        raise SystemExit(
            "Failed to import masked_imputation26. Ensure dependencies (torch, numpy) are available.\n"
            f"Python: {sys.executable}\n"
            f"Error: {exc}\n"
        ) from exc
    return mi26


def _counts_obs_from_logcounts(logcounts: np.ndarray, counts: Optional[np.ndarray]) -> np.ndarray:
    if counts is None:
        return np.clip(np.expm1(logcounts * np.log(2.0)), 0.0, None).astype(np.float32)
    return np.clip(counts, 0.0, None).astype(np.float32)


def run_maskimpute(
    logcounts: np.ndarray,
    counts: Optional[np.ndarray],
    *,
    seed: int,
    bio_reg_weight: Optional[float] = None,
) -> np.ndarray:
    import torch

    mi26 = _import_maskimpute_impl()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    counts_obs = _counts_obs_from_logcounts(logcounts, counts)
    zeros_obs = counts_obs <= 0.0
    counts_max = counts_obs.max(axis=0)

    if bio_reg_weight is not None:
        mi26.AE_PARAMS["bio_reg_weight"] = float(bio_reg_weight)
    mi26.set_seed(int(seed))
    p_bio = mi26.splat_cellaware_bio_prob(
        counts=counts_obs,
        zeros_obs=zeros_obs,
        disp_mode=mi26.BIO_PARAMS["disp_mode"],
        use_cell_factor=mi26.BIO_PARAMS["use_cell_factor"],
    )

    log_recon = mi26.train_autoencoder_reconstruct(
        logcounts=logcounts,
        counts_max=counts_max,
        p_bio=p_bio,
        device=device,
        fast_mode=True,
        amp_enabled=True,
        compile_enabled=True,
        fast_batch_mult=2,
        num_workers=2,
    )
    p_refined = mi26.refine_bio_prob_with_reconstruction(
        recon_log=log_recon,
        counts_obs=counts_obs,
        zeros_obs=zeros_obs,
    )
    log_imputed = mi26.apply_zero_gate(log_recon, p_refined, zeros_obs)
    return log_imputed


def parse_methods(raw: Optional[str]) -> List[str]:
    if not raw:
        return list(METHODS)
    if raw.lower() == "all":
        return list(METHODS)
    methods = [METHOD_ALIASES.get(m.strip().lower(), m.strip().lower()) for m in raw.split(",") if m.strip()]
    unknown = sorted(set(methods) - set(METHODS))
    if unknown:
        raise ValueError(f"Unknown methods: {', '.join(unknown)}. Allowed: {METHOD_HELP} or 'all'.")
    return list(dict.fromkeys(methods))


def _write_method_table(output_dir: Path, method: str, rows: List[Dict[str, object]]) -> None:
    out_path = output_dir / f"{method}_mse_table.tsv"
    columns = [
        "dataset",
        "mse",
        "mse_std",
        "mse_dropout",
        "mse_dropout_std",
        "mse_biozero",
        "mse_biozero_std",
        "mse_non_zero",
        "mse_non_zero_std",
        "mse_marker",
        "mse_marker_std",
        "mae",
        "mae_std",
        "mae_dropout",
        "mae_dropout_std",
        "mae_biozero",
        "mae_biozero_std",
        "mae_non_zero",
        "mae_non_zero_std",
        "mae_marker",
        "mae_marker_std",
        "gnrmse",
        "gnrmse_std",
        "gnrmse_marker",
        "gnrmse_marker_std",
        "corr_err",
        "corr_err_std",
        "n_corr_genes",
        "n_corr_genes_std",
        "runtime_sec",
        "runtime_sec_std",
        "n_repeats",
        "n_total",
        "n_dropout",
        "n_biozero",
        "n_non_zero",
        "n_marker",
        "n_marker_genes",
        "error",
    ]

    with out_path.open("w", encoding="utf-8") as f:
        f.write("\t".join(columns) + "\n")
        for row in sorted(rows, key=lambda r: str(r.get("dataset", ""))):
            f.write("\t".join(str(row.get(col, "")) for col in columns) + "\n")

    print(f"Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run imputation methods and report benchmark metrics for .rds datasets.")
    parser.add_argument("input_path", help="Path to .rds file or directory")
    parser.add_argument("output_dir", help="Output directory for <method>_mse_table.tsv")
    parser.add_argument(
        "--methods",
        default=None,
        help=f"Comma-separated list ({METHOD_HELP}) or 'all'.",
    )
    parser.add_argument(
        "methods_arg",
        nargs="?",
        default=None,
        help=f"Optional methods list ({METHOD_HELP}) or 'all'.",
    )
    parser.add_argument("--n-jobs", type=int, default=1, help="MAGIC n_jobs value")
    parser.add_argument("--n-repeat", type=int, default=5, help="Number of repeats per method.")

    g_dca = parser.add_argument_group("DCA Options")
    g_dca.add_argument("--dca-bin", type=str, default="~/miniconda3/envs/dca_env/bin/dca")
    g_dca.add_argument(
        "--dca-ae-type",
        type=str,
        default=None,
        help="Override DCA --type (default: nb-conddisp). Omit to use DCA defaults.",
    )
    g_dca.add_argument(
        "--dca-epochs",
        type=int,
        default=None,
        help="Override DCA --epochs (default: 300). Omit to use DCA defaults.",
    )
    g_dca.add_argument(
        "--dca-batch-size",
        type=int,
        default=None,
        help="Override DCA --batchsize (default: 32). Omit to use DCA defaults.",
    )
    g_dca.add_argument(
        "--dca-threads",
        type=int,
        default=None,
        help="Override DCA --threads (default: all cores). Omit to use DCA defaults.",
    )
    g_dca.add_argument(
        "--dca-ridge",
        type=float,
        default=None,
        help="Override DCA --ridge (default: 0.0). Omit to use DCA defaults.",
    )
    g_dca.add_argument("--dca-verbose", action="store_true")
    g_dca.add_argument(
        "--dca-timeout-sec",
        type=float,
        default=None,
        help="Optional wall-clock timeout for one DCA run.",
    )

    g_scvi = parser.add_argument_group("scVI Options")
    g_scvi.add_argument("--scvi-max-epochs", type=int, default=400)
    g_scvi.add_argument("--scvi-latent", type=int, default=10)
    g_scvi.add_argument("--scvi-hidden", type=int, default=128)
    g_scvi.add_argument("--scvi-layers", type=int, default=2)
    g_scvi.add_argument("--scvi-batch-size", type=int, default=256)

    g_alra = parser.add_argument_group("ALRA Options")
    g_alra.add_argument("--alra-rank", type=int, default=0, help="Fixed ALRA rank; 0 selects rank automatically.")
    g_alra.add_argument("--alra-max-rank", type=int, default=100)
    g_alra.add_argument("--alra-n-iter", type=int, default=7)

    g_auto = parser.add_argument_group("AutoClass Options")
    g_auto.add_argument(
        "--autoclass-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "AutoClass"),
        help="Path to AutoClass repo clone (default: ./AutoClass)",
    )
    g_auto.add_argument(
        "--autoclass-kwargs",
        type=str,
        default="",
        help="Optional comma-separated key=value pairs passed to AutoClass",
    )

    args = parser.parse_args()
    raw_methods = args.methods if args.methods is not None else args.methods_arg
    methods = parse_methods(raw_methods)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, List[Dict[str, object]]] = {m: [] for m in methods}
    autoclass_kwargs = _parse_kv_pairs(args.autoclass_kwargs) if "autoclass" in methods else {}

    for path in collect_rds_files(args.input_path):
        ds_name = dataset_name_from_path(path)
        print(f"\n=== {ds_name} ===")
        dataset = load_dataset(str(path))
        if dataset is None:
            print(f"[WARN] {ds_name}: missing logTrueCounts; skipping.")
            continue

        logcounts = dataset["logcounts"]
        log_true = dataset["log_true"]
        counts = dataset["counts"]
        norm_info = dataset["norm"]
        marker_gene_mask = dataset["marker_gene_mask"]
        masks = compute_masks(log_true, logcounts, marker_gene_mask=marker_gene_mask)
        mask_counts = compute_mask_counts(masks)

        for method in methods:
            display_name = METHOD_DISPLAY_NAMES.get(method, method)
            print(f"  -> {display_name} [{method}]")
            row: Dict[str, object] = {"dataset": ds_name}
            metrics_runs: List[Dict[str, float]] = []
            runtimes: List[float] = []
            error_msg = ""

            for rep in range(int(args.n_repeat)):
                try:
                    t0 = time.perf_counter()
                    if method == "magic":
                        log_imp = run_magic(logcounts, args.n_jobs)
                    elif method == "dca":
                        if counts is None:
                            raise RuntimeError("counts assay not available; cannot run DCA.")
                        counts_imp = run_dca(
                            counts,
                            dca_bin=args.dca_bin,
                            ae_type=args.dca_ae_type,
                            epochs=args.dca_epochs,
                            batch_size=args.dca_batch_size,
                            threads=args.dca_threads,
                            ridge=args.dca_ridge,
                            verbose=args.dca_verbose,
                            timeout_sec=args.dca_timeout_sec,
                        )
                        log_imp = normalize_counts_to_logcounts(
                            counts_imp, float(norm_info["target_sum"])
                        )
                    elif method == "scvi":
                        if counts is None:
                            raise RuntimeError("counts assay not available; cannot run scVI.")
                        log_imp = run_scvi(
                            counts,
                            float(norm_info["target_sum"]),
                            seed=42 + rep,
                            max_epochs=args.scvi_max_epochs,
                            n_latent=args.scvi_latent,
                            n_hidden=args.scvi_hidden,
                            n_layers=args.scvi_layers,
                            batch_size=args.scvi_batch_size,
                        )
                    elif method == "alra":
                        log_imp = run_alra(
                            logcounts,
                            seed=42 + rep,
                            rank=args.alra_rank if args.alra_rank and args.alra_rank > 0 else None,
                            max_rank=args.alra_max_rank,
                            n_iter=args.alra_n_iter,
                        )
                    elif method == "autoclass":
                        log_imp = run_autoclass(logcounts, args.autoclass_dir, autoclass_kwargs)
                    elif method == "balanced_mse":
                        log_imp = run_maskimpute(
                            logcounts,
                            counts,
                            bio_reg_weight=None,
                            seed=42 + rep,
                        )
                    else:
                        raise RuntimeError(f"Unsupported method: {method}")
                    runtime = time.perf_counter() - t0

                    if log_imp.shape != log_true.shape:
                        raise ValueError(
                            f"{ds_name}: {method} output shape {log_imp.shape} does not match logTrueCounts {log_true.shape}"
                        )

                    metrics_runs.append(compute_error_metrics(log_imp, log_true, masks))
                    runtimes.append(runtime)
                except Exception as exc:
                    error_msg = str(exc)
                    print(f"    [ERROR] {method} failed: {exc}")
                    break

            def _mean_std(values: Sequence[float]) -> Tuple[float, float]:
                arr = np.asarray(list(values), dtype=float)
                if arr.size == 0:
                    return float("nan"), float("nan")
                mean = float(np.mean(arr))
                std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
                return mean, std

            if metrics_runs:
                metric_keys = (
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
                    "n_corr_genes",
                )
                for key in metric_keys:
                    vals = [m[key] for m in metrics_runs]
                    mean, std = _mean_std(vals)
                    row[key] = mean
                    row[f"{key}_std"] = std

                runtime_mean, runtime_std = _mean_std(runtimes)
                row["runtime_sec"] = runtime_mean
                row["runtime_sec_std"] = runtime_std
                row["n_repeats"] = len(metrics_runs)
                row.update(mask_counts)
                row["error"] = error_msg
            else:
                row.update(
                    {
                        "mse": float("nan"),
                        "mse_std": float("nan"),
                        "mse_dropout": float("nan"),
                        "mse_dropout_std": float("nan"),
                        "mse_biozero": float("nan"),
                        "mse_biozero_std": float("nan"),
                        "mse_non_zero": float("nan"),
                        "mse_non_zero_std": float("nan"),
                        "mse_marker": float("nan"),
                        "mse_marker_std": float("nan"),
                        "mae": float("nan"),
                        "mae_std": float("nan"),
                        "mae_dropout": float("nan"),
                        "mae_dropout_std": float("nan"),
                        "mae_biozero": float("nan"),
                        "mae_biozero_std": float("nan"),
                        "mae_non_zero": float("nan"),
                        "mae_non_zero_std": float("nan"),
                        "mae_marker": float("nan"),
                        "mae_marker_std": float("nan"),
                        "gnrmse": float("nan"),
                        "gnrmse_std": float("nan"),
                        "gnrmse_marker": float("nan"),
                        "gnrmse_marker_std": float("nan"),
                        "corr_err": float("nan"),
                        "corr_err_std": float("nan"),
                        "n_corr_genes": float("nan"),
                        "n_corr_genes_std": float("nan"),
                        "runtime_sec": float("nan"),
                        "runtime_sec_std": float("nan"),
                        "n_repeats": 0,
                        **mask_counts,
                        "error": error_msg or "No successful repeats.",
                    }
                )

            results[method].append(row)

    for method, rows in results.items():
        if not rows:
            print(f"No datasets processed for method '{method}'.")
            continue
        _write_method_table(output_dir, method, rows)


if __name__ == "__main__":
    main()
