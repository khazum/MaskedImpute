#!/usr/bin/env python3
"""
run_imputation.py
----------------

Runs multiple imputation methods (MAGIC, legacy DCA, AutoClass) on
SingleCellExperiment .rds files and reports MSEs in log2(1+normalized) space
vs. logTrueCounts, matching experiment2.py dataset conventions.

Required Python packages: rds2py, numpy.
Method-specific dependencies: magic-impute (MAGIC), pandas + DCA CLI (DCA bridge),
and AutoClass with its Python requirements (e.g., tensorflow).

Normalization for imputed counts uses the dataset's TrueCounts-derived size
factors (libSizeTrueCounts/scaleFactorTrueCounts or stored sizeFactor values).
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
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from rds2py import read_rds
except Exception as exc:  # pragma: no cover - import error surfaced for user
    raise SystemExit(
        "Failed to import rds2py in this Python. See script header for requirements.\n"
        f"Python: {sys.executable}\n"
        f"Error: {exc}\n"
    ) from exc

EPSILON = 1e-6
METHODS = ("magic", "dca", "autoclass", "low_mse", "balanced_mse")


def _masked_mse(diff: np.ndarray, mask: np.ndarray) -> float:
    n = int(mask.sum())
    if n == 0:
        return float("nan")
    return float(np.mean((diff[mask]) ** 2))


def compute_masks(log_true: np.ndarray, log_obs: np.ndarray) -> Dict[str, np.ndarray]:
    return {
        "biozero": log_true <= EPSILON,
        "dropout": (log_true > EPSILON) & (log_obs <= EPSILON),
        "non_zero": (log_true > EPSILON) & (log_obs > EPSILON),
    }


def compute_mse_metrics(
    log_imp: np.ndarray, log_true: np.ndarray, masks: Dict[str, np.ndarray]
) -> Dict[str, float]:
    if log_imp.shape != log_true.shape:
        raise ValueError(f"log_imp shape {log_imp.shape} does not match log_true {log_true.shape}.")
    if masks["biozero"].shape != log_true.shape:
        raise ValueError("Mask shape does not match log_true.")
    diff = log_true - log_imp
    return {
        "mse": float(np.mean(diff ** 2)),
        "mse_dropout": _masked_mse(diff, masks["dropout"]),
        "mse_biozero": _masked_mse(diff, masks["biozero"]),
        "mse_non_zero": _masked_mse(diff, masks["non_zero"]),
        "n_total": int(diff.size),
        "n_dropout": int(masks["dropout"].sum()),
        "n_biozero": int(masks["biozero"].sum()),
        "n_non_zero": int(masks["non_zero"].sum()),
    }


def compute_mask_counts(masks: Dict[str, np.ndarray]) -> Dict[str, int]:
    return {
        "n_total": int(masks["biozero"].size),
        "n_dropout": int(masks["dropout"].sum()),
        "n_biozero": int(masks["biozero"].sum()),
        "n_non_zero": int(masks["non_zero"].sum()),
    }


def normalize_counts_to_logcounts(
    counts_mat: np.ndarray,
    size_factors: np.ndarray,
) -> np.ndarray:
    """Normalize counts to log2(1+normalized) using TrueCounts size factors."""
    x = np.asarray(counts_mat, dtype=np.float64)
    x = np.clip(x, 0.0, None)
    sf = np.asarray(size_factors, dtype=np.float64)
    sf = np.where(sf > 0.0, sf, 1.0)
    if sf.shape[0] != x.shape[0]:
        raise ValueError(f"Normalization size mismatch: size_factors {sf.shape[0]} vs counts {x.shape[0]}.")
    norm = x / sf[:, None]
    return np.log2(1.0 + norm).astype(np.float32)


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


def _get_metadata_normalization(sce) -> Dict[str, object]:
    md = getattr(sce, "metadata", None)
    if isinstance(md, dict):
        norm = md.get("normalization")
        if isinstance(norm, dict):
            return norm
    return {}


def get_normalization_info(sce) -> Dict[str, np.ndarray]:
    """Return TrueCounts-based library sizes, size factors, and scale factor."""
    md_norm = _get_metadata_normalization(sce)

    lib_true = _get_coldata_column(sce, "libSizeTrueCounts")
    if lib_true is None and md_norm.get("library_sizes") is not None:
        lib_true = md_norm.get("library_sizes")
    if lib_true is None:
        try:
            lib_true = np.sum(sce.assay("TrueCounts"), axis=0)
        except Exception as exc:
            raise RuntimeError("Missing TrueCounts or libSizeTrueCounts for normalization.") from exc

    lib_true = np.asarray(lib_true, dtype=np.float64)

    scale_vals = _get_coldata_column(sce, "scaleFactorTrueCounts")
    scale_factor = None
    if scale_vals is not None:
        scale_vals = np.asarray(scale_vals, dtype=np.float64)
        scale_vals = scale_vals[np.isfinite(scale_vals) & (scale_vals > 0.0)]
        if scale_vals.size:
            scale_factor = float(scale_vals[0])
    if scale_factor is None and md_norm.get("scale_factor") is not None:
        try:
            scale_factor = float(md_norm.get("scale_factor"))
        except Exception:
            scale_factor = None
    if scale_factor is None:
        finite = lib_true[lib_true > 0.0]
        scale_factor = float(np.median(finite)) if finite.size else 1.0
    if not np.isfinite(scale_factor) or scale_factor <= 0.0:
        scale_factor = 1.0

    size_factors = None
    for key in ("sizeFactorTrueCounts", "sizeFactorsTrueCounts"):
        size_factors = _get_coldata_column(sce, key)
        if size_factors is not None:
            break
    if size_factors is None and md_norm.get("size_factors") is not None:
        size_factors = md_norm.get("size_factors")
    if size_factors is None:
        size_factors = lib_true / scale_factor

    size_factors = np.asarray(size_factors, dtype=np.float64)
    size_factors = np.where(np.isfinite(size_factors) & (size_factors > 0.0), size_factors, 1.0)

    return {
        "lib_true": lib_true,
        "scale_factor": scale_factor,
        "size_factors": size_factors,
    }


def load_dataset(path: str) -> Optional[Dict[str, np.ndarray]]:
    sce = read_rds(path)
    if not hasattr(sce, "assay"):
        raise TypeError(f"Unsupported RDS object (expected SingleCellExperiment): {type(sce)}")

    logcounts = sce.assay("logcounts").T.astype("float32")

    log_true = None
    for assay_name in ("logTrueCounts", "perfect_logcounts"):
        try:
            log_true = sce.assay(assay_name).T.astype("float32")
            break
        except Exception:
            continue

    if log_true is None:
        return None

    counts = None
    try:
        counts = sce.assay("counts").T.astype("float32")
    except Exception:
        counts = None

    norm_info = get_normalization_info(sce)

    if logcounts.shape != log_true.shape:
        raise ValueError(f"Assay shape mismatch: logcounts {logcounts.shape} vs logTrueCounts {log_true.shape}.")
    if counts is not None and counts.shape != logcounts.shape:
        raise ValueError(f"Assay shape mismatch: counts {counts.shape} vs logcounts {logcounts.shape}.")
    if norm_info["size_factors"].shape[0] != logcounts.shape[0]:
        raise ValueError(
            f"Normalization size mismatch: size_factors {norm_info['size_factors'].shape[0]} vs cells {logcounts.shape[0]}."
        )

    return {
        "logcounts": logcounts,
        "log_true": log_true,
        "counts": counts,
        "norm": norm_info,
    }


def collect_rds_files(input_path: str) -> List[Path]:
    path = Path(input_path)
    if path.is_dir():
        return sorted(path.rglob("*.rds"))
    if path.is_file():
        return [path]
    raise FileNotFoundError(f"Input path not found: {input_path}")


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

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_dir_path = Path(tmpdir)
        input_file = tmp_dir_path / "matrix.csv"
        output_dir = tmp_dir_path / "dca_out"

        try:
            gene_names = [f"gene_{i}" for i in range(C_work.shape[1])]
            cell_names = [f"cell_{i}" for i in range(C_work.shape[0])]
            df = pd.DataFrame(C_work.T, index=gene_names, columns=cell_names)
            df.to_csv(input_file, sep=",")
        except Exception as e:
            raise RuntimeError(f"Failed to write input CSV for DCA: {e}")

        cmd = [dca_bin, str(input_file), str(output_dir)]

        if ae_type is not None:
            cmd.extend(["--type", ae_type])
        if threads is not None:
            cmd.extend(["--threads", str(threads)])
        if epochs is not None:
            cmd.extend(["-e", str(epochs)])
        if batch_size is not None:
            cmd.extend(["-b", str(batch_size)])
        if ridge is not None:
            cmd.extend(["--ridge", str(ridge)])

        try:
            subprocess.run(
                cmd,
                check=True,
                stdout=None if verbose else subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
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


def _import_masked26():
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


def _cell_zero_norm(zeros_obs: np.ndarray) -> np.ndarray:
    cell_zero_frac = zeros_obs.mean(axis=1).astype(np.float32)
    cz_lo = float(np.percentile(cell_zero_frac, 5.0))
    cz_hi = float(np.percentile(cell_zero_frac, 95.0))
    cz_span = max(cz_hi - cz_lo, EPSILON)
    return np.clip((cell_zero_frac - cz_lo) / cz_span, 0.0, 1.0).astype(np.float32)


def run_masked26(
    logcounts: np.ndarray,
    counts: Optional[np.ndarray],
    *,
    bio_reg_weight: float,
    seed: int,
) -> np.ndarray:
    import torch

    mi26 = _import_masked26()
    device = torch.device("cuda")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for masked_imputation26 but not available.")

    counts_obs = _counts_obs_from_logcounts(logcounts, counts)
    zeros_obs = counts_obs <= 0.0
    counts_max = counts_obs.max(axis=0)
    cell_zero_norm = _cell_zero_norm(zeros_obs)

    mi26.AE_PARAMS["bio_reg_weight"] = float(bio_reg_weight)
    mi26.set_seed(int(seed))
    p_bio = mi26.splat_cellaware_bio_prob(
        counts=counts_obs,
        zeros_obs=zeros_obs,
        disp_mode=mi26.BIO_PARAMS["disp_mode"],
        use_cell_factor=mi26.BIO_PARAMS["use_cell_factor"],
    )
    if float(mi26.BIO_PARAMS["cell_zero_weight"]) > 0.0:
        cell_w = np.clip(
            float(mi26.BIO_PARAMS["cell_zero_weight"]) * cell_zero_norm, 0.0, 1.0
        )
        p_bio = p_bio * (1.0 - cell_w[:, None])

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

    log_imputed = log_recon.copy()
    log_imputed[~zeros_obs] = logcounts[~zeros_obs]
    return log_imputed


def parse_methods(raw: Optional[str]) -> List[str]:
    if not raw:
        return list(METHODS)
    if raw.lower() == "all":
        return list(METHODS)
    methods = [m.strip().lower() for m in raw.split(",") if m.strip()]
    unknown = sorted(set(methods) - set(METHODS))
    if unknown:
        raise ValueError(f"Unknown methods: {', '.join(unknown)}. Allowed: {', '.join(METHODS)} or 'all'.")
    return methods


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
        "runtime_sec",
        "runtime_sec_std",
        "n_repeats",
        "n_total",
        "n_dropout",
        "n_biozero",
        "n_non_zero",
        "error",
    ]

    with out_path.open("w", encoding="utf-8") as f:
        f.write("\t".join(columns) + "\n")
        for row in sorted(rows, key=lambda r: str(r.get("dataset", ""))):
            f.write("\t".join(str(row.get(col, "")) for col in columns) + "\n")

    print(f"Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run imputation methods and report MSE metrics for .rds datasets.")
    parser.add_argument("input_path", help="Path to .rds file or directory")
    parser.add_argument("output_dir", help="Output directory for <method>_mse_table.tsv")
    parser.add_argument(
        "--methods",
        default=None,
        help="Comma-separated list (magic,dca,autoclass,low_mse,balanced_mse) or 'all'.",
    )
    parser.add_argument(
        "methods_arg",
        nargs="?",
        default=None,
        help="Optional methods list (magic,dca,autoclass,low_mse,balanced_mse) or 'all'.",
    )
    parser.add_argument("--n-jobs", type=int, default=1, help="MAGIC n_jobs value")
    parser.add_argument("--n-repeat", type=int, default=10, help="Number of repeats per method.")

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
        ds_name = path.stem
        print(f"\n=== {ds_name} ===")
        dataset = load_dataset(str(path))
        if dataset is None:
            print(f"[WARN] {ds_name}: missing logTrueCounts; skipping.")
            continue

        logcounts = dataset["logcounts"]
        log_true = dataset["log_true"]
        counts = dataset["counts"]
        norm_info = dataset["norm"]
        masks = compute_masks(log_true, logcounts)
        mask_counts = compute_mask_counts(masks)

        for method in methods:
            print(f"  -> {method}")
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
                        )
                        log_imp = normalize_counts_to_logcounts(
                            counts_imp, norm_info["size_factors"]
                        )
                    elif method == "autoclass":
                        log_imp = run_autoclass(logcounts, args.autoclass_dir, autoclass_kwargs)
                    elif method in ("low_mse", "balanced_mse"):
                        bio_reg = 0.0 if method == "low_mse" else 1.0
                        log_imp = run_masked26(
                            logcounts,
                            counts,
                            bio_reg_weight=bio_reg,
                            seed=42 + rep,
                        )
                    else:
                        raise RuntimeError(f"Unsupported method: {method}")
                    runtime = time.perf_counter() - t0

                    if log_imp.shape != log_true.shape:
                        raise ValueError(
                            f"{ds_name}: {method} output shape {log_imp.shape} does not match logTrueCounts {log_true.shape}"
                        )

                    metrics_runs.append(compute_mse_metrics(log_imp, log_true, masks))
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
                for key in ("mse", "mse_dropout", "mse_biozero", "mse_non_zero"):
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
