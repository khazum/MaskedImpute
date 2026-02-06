#!/usr/bin/env python3
"""
run_clustering.py
----------------

Run clustering evaluation (ARI, NMI, Purity, ASW) on SingleCellExperiment .rds
files for multiple imputation methods. Methods mirror run_imputation.py and
add the experiment autoencoder: magic, dca, autoclass, low_mse, balanced_mse,
experiment.

Procedure (per method):
- impute (or baseline) to obtain logcounts
- PCA to at most 50 components
- k-means with k = number of unique labels (min 2)
- metrics: ARI, NMI, Purity, ASW
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from clustering_eval import evaluate_clustering, _adjusted_rand_score, _normalized_mutual_info, _purity_score, _silhouette_score

try:
    from rds2py import read_rds
except Exception as exc:
    raise SystemExit(
        "Failed to import rds2py. Install requirements or run in the proper env.\n"
        f"Error: {exc}"
    ) from exc

try:
    import run_imputation as imp
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Failed to import run_imputation.py. Ensure it is on PYTHONPATH.\n"
        f"Error: {exc}"
    ) from exc

METHODS = ("magic", "dca", "autoclass", "low_mse", "balanced_mse", "experiment")


def _counts_obs_from_logcounts(logcounts: np.ndarray, counts: Optional[np.ndarray]) -> np.ndarray:
    if counts is None:
        return np.clip(np.expm1(logcounts * np.log(2.0)), 0.0, None).astype(np.float32)
    return np.clip(counts, 0.0, None).astype(np.float32)


def _cell_zero_norm(zeros_obs: np.ndarray) -> np.ndarray:
    cell_zero_frac = zeros_obs.mean(axis=1).astype(np.float32)
    cz_lo = float(np.percentile(cell_zero_frac, 5.0))
    cz_hi = float(np.percentile(cell_zero_frac, 95.0))
    cz_span = max(cz_hi - cz_lo, imp.EPSILON)
    return np.clip((cell_zero_frac - cz_lo) / cz_span, 0.0, 1.0).astype(np.float32)


def _run_masked26_clustering(
    logcounts: np.ndarray,
    counts: Optional[np.ndarray],
    *,
    bio_reg_weight: float,
    seed: int,
    device: str,
    output: str,
    latent_norm: str,
    blend_alpha: float,
    recon_zscore: bool,
    keep_nonzero: bool,
    add_cellstats: bool,
    cellstats_zscore: bool,
    refine_k_labels: Optional[int],
    config: Optional[Dict[str, object]] = None,
) -> np.ndarray:
    """Run masked_imputation26 without preserving observed nonzeros (clustering only)."""
    mi26 = imp._import_masked26()
    if config:
        mi26.apply_config(config)
    if refine_k_labels is not None and bool(mi26.AE_PARAMS.get("cluster_refine", False)):
        if int(mi26.AE_PARAMS.get("cluster_refine_k", 0)) <= 1:
            mi26.AE_PARAMS["cluster_refine_k"] = int(refine_k_labels)
    device = imp._resolve_device(device)

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

    use_fast = device.type == "cuda"
    log_recon = mi26.train_autoencoder_reconstruct(
        logcounts=logcounts,
        counts_max=counts_max,
        p_bio=p_bio,
        device=device,
        fast_mode=use_fast,
        amp_enabled=use_fast,
        compile_enabled=use_fast,
        fast_batch_mult=2 if use_fast else 1,
        num_workers=2 if use_fast else 0,
        output=output,
        latent_norm=latent_norm,
    )

    # NOTE: for clustering we do NOT preserve observed nonzeros by default
    out = log_recon.astype(np.float32, copy=False)
    if keep_nonzero:
        out = out.copy()
        out[~zeros_obs] = logcounts[~zeros_obs]
    out_mode = str(output or "recon").strip().lower()
    if out_mode in ("recon", "recon+latent"):
        n_genes = logcounts.shape[1]
        recon = out[:, :n_genes] if out_mode == "recon+latent" else out
        alpha = float(blend_alpha)
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("masked26_blend_alpha must be in [0,1].")
        if alpha < 1.0:
            recon = alpha * recon + (1.0 - alpha) * logcounts
        if recon_zscore:
            mean = recon.mean(axis=0, keepdims=True)
            std = recon.std(axis=0, keepdims=True)
            std = np.where(std > imp.EPSILON, std, 1.0)
            recon = (recon - mean) / std
        if out_mode == "recon":
            out = recon.astype(np.float32, copy=False)
        else:
            latent = out[:, n_genes:]
            out = np.concatenate([recon.astype(np.float32, copy=False), latent], axis=1)
    if add_cellstats:
        lib_size = counts_obs.sum(axis=1).astype(np.float32)
        lib_size = np.log1p(lib_size)
        cell_zero_norm = _cell_zero_norm(zeros_obs).astype(np.float32)
        stats = np.stack([lib_size, cell_zero_norm], axis=1)
        if cellstats_zscore:
            mean = stats.mean(axis=0, keepdims=True)
            std = stats.std(axis=0, keepdims=True)
            std = np.where(std > imp.EPSILON, std, 1.0)
            stats = (stats - mean) / std
        out = np.concatenate([out, stats], axis=1)
    return out.astype(np.float32, copy=False)


LABEL_KEYS = ("cell_type1", "labels", "Group", "label")


def _pca_embed(X: np.ndarray, n_components: int) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    X = X - X.mean(axis=0, keepdims=True)
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    return U[:, :n_components] * S[:n_components]


def _knn_smooth(X: np.ndarray, k: int, steps: int, pca_components: int) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if steps <= 0 or k <= 0:
        return X
    n = X.shape[0]
    k = min(k, max(1, n - 1))
    for _ in range(int(steps)):
        emb = _pca_embed(X, min(int(pca_components), X.shape[1], n))
        sq = np.sum(emb * emb, axis=1, keepdims=True)
        dist2 = sq + sq.T - 2.0 * (emb @ emb.T)
        dist2 = np.maximum(dist2, 0.0)
        # argsort to get k nearest neighbors (excluding self at idx 0)
        nn_idx = np.argsort(dist2, axis=1)[:, 1 : k + 1]
        X_new = np.empty_like(X)
        for i in range(n):
            X_new[i] = X[nn_idx[i]].mean(axis=0)
        X = X_new
    return X


def _metrics_from_labels(
    X: np.ndarray,
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    n_components: int,
    emb: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    if emb is None:
        emb = _pca_embed(X, min(int(n_components), X.shape[1], X.shape[0]))
    ch = float("nan")
    db = float("nan")
    try:
        if len(np.unique(pred_labels)) >= 2:
            from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score

            ch = float(calinski_harabasz_score(emb, pred_labels))
            db = float(davies_bouldin_score(emb, pred_labels))
    except Exception:
        pass
    return {
        "ASW": round(_silhouette_score(emb, pred_labels), 4),
        "ARI": round(_adjusted_rand_score(true_labels, pred_labels), 4),
        "NMI": round(_normalized_mutual_info(true_labels, pred_labels), 4),
        "PS": round(_purity_score(true_labels, pred_labels), 4),
        "CH": ch,
        "DB": db,
    }


def _cluster_labels(
    X: np.ndarray,
    labels: np.ndarray,
    *,
    cluster: str,
    embed: str,
    pca_components: int,
    tsne_dim: int,
    tsne_perplexity: float,
    tsne_iter: int,
    kmeans_init: int,
    kmeans_iter: int,
    spectral_k: int,
    gmm_cov: str,
    gmm_reg: float,
    agg_linkage: str,
    seed: int,
) -> Dict[str, float]:
    embed_mode = str(embed or "pca").strip().lower()
    if embed_mode == "pca":
        emb = _pca_embed(X, min(int(pca_components), X.shape[1], X.shape[0]))
    elif embed_mode == "tsne":
        from sklearn.manifold import TSNE

        n_samples = X.shape[0]
        perp = float(tsne_perplexity)
        perp = max(2.0, min(perp, float(max(2, n_samples - 1))))
        emb = TSNE(
            n_components=int(tsne_dim),
            perplexity=perp,
            n_iter=int(tsne_iter),
            init="pca",
            learning_rate="auto",
            random_state=seed,
        ).fit_transform(X)
    else:
        raise ValueError(f"Unknown embed mode: {embed_mode}")

    if cluster == "kmeans":
        from sklearn.cluster import KMeans

        km = KMeans(
            n_clusters=max(2, len(np.unique(labels))),
            n_init=int(kmeans_init),
            max_iter=int(kmeans_iter),
            random_state=seed,
        )
        pred = km.fit_predict(emb)
        return _metrics_from_labels(X, labels, pred, int(pca_components), emb=emb)
    k = max(2, len(np.unique(labels)))
    if cluster == "spectral":
        from sklearn.cluster import SpectralClustering

        pred = SpectralClustering(
            n_clusters=int(k),
            n_neighbors=int(spectral_k),
            affinity="nearest_neighbors",
            assign_labels="kmeans",
            random_state=seed,
        ).fit_predict(emb)
    elif cluster == "gmm":
        from sklearn.mixture import GaussianMixture

        gm = GaussianMixture(
            n_components=int(k),
            covariance_type=str(gmm_cov),
            reg_covar=float(gmm_reg),
            random_state=seed,
        )
        pred = gm.fit_predict(emb)
    elif cluster == "agglomerative":
        from sklearn.cluster import AgglomerativeClustering

        pred = AgglomerativeClustering(
            n_clusters=int(k),
            linkage=str(agg_linkage),
        ).fit_predict(emb)
    else:
        raise ValueError(f"Unknown cluster method: {cluster}")
    return _metrics_from_labels(X, labels, pred, int(pca_components), emb=emb)


def _latent_dim_from_config(config: Optional[Dict[str, object]]) -> Optional[int]:
    if isinstance(config, dict):
        model_params = config.get("model_params")
        if isinstance(model_params, dict) and "bottleneck" in model_params:
            try:
                return int(model_params["bottleneck"])
            except Exception:
                return None
    try:
        mi26 = imp._import_masked26()
        return int(mi26.MODEL_PARAMS.get("bottleneck", 0))
    except Exception:
        return None


def _extract_labels(sce) -> Tuple[np.ndarray, str]:
    colmd = getattr(sce, "column_data", None) or getattr(sce, "colData", None)
    y = None
    source = None
    if colmd is not None:
        if hasattr(colmd, "get_column_names") and hasattr(colmd, "get_column"):
            colnames = list(map(str, colmd.get_column_names()))
            for key in LABEL_KEYS:
                if key in colnames:
                    y = np.asarray(colmd.get_column(key))
                    source = key
                    break
        elif hasattr(colmd, "columns"):
            colnames = list(map(str, getattr(colmd, "columns", [])))
            for key in LABEL_KEYS:
                if key in colnames:
                    y = np.asarray(colmd[key])
                    source = key
                    break
        elif isinstance(colmd, dict):
            for key in LABEL_KEYS:
                if key in colmd:
                    y = np.asarray(colmd[key])
                    source = key
                    break

    if y is None:
        raise RuntimeError(f"No label column found. Tried: {', '.join(LABEL_KEYS)}")

    # factorize labels to integer ids
    uniq, labels = np.unique(np.asarray(y), return_inverse=True)
    labels = labels.astype(int)
    return labels, source or "unknown"


def load_dataset(path: str) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, np.ndarray], np.ndarray, str]:
    sce = read_rds(path)
    if not hasattr(sce, "assay"):
        raise TypeError(f"Unsupported RDS object (expected SingleCellExperiment): {type(sce)}")
    logcounts = sce.assay("logcounts").T.astype("float32")
    counts = None
    try:
        counts = sce.assay("counts").T.astype("float32")
    except Exception:
        counts = None
    try:
        norm = imp.get_normalization_info(sce)
    except Exception:
        norm = {"size_factors": None}
    labels, source = _extract_labels(sce)
    if logcounts.shape[0] != labels.shape[0]:
        raise ValueError(f"Cells mismatch: logcounts {logcounts.shape[0]} vs labels {labels.shape[0]}")
    return logcounts, counts, norm, labels, source


def parse_methods(raw: Optional[str]) -> List[str]:
    if not raw or raw.lower() == "all":
        return list(METHODS)
    methods = [m.strip().lower() for m in raw.split(",") if m.strip()]
    unknown = sorted(set(methods) - set(METHODS))
    if unknown:
        raise ValueError(f"Unknown methods: {', '.join(unknown)}. Allowed: {', '.join(METHODS)} or 'all'.")
    return methods


def _run_masked26_ensemble(
    logcounts: np.ndarray,
    counts: Optional[np.ndarray],
    *,
    bio_reg: float,
    seed: int,
    args: argparse.Namespace,
    labels: np.ndarray,
    config_list: List[Dict[str, object]],
    weights: Optional[List[float]] = None,
) -> np.ndarray:
    outputs: List[np.ndarray] = []
    for cfg in config_list:
        outputs.append(
            _run_masked26_clustering(
                logcounts,
                counts,
                bio_reg_weight=bio_reg,
                seed=seed,
                device=args.masked26_device,
                output=args.masked26_output,
                latent_norm=args.masked26_latent_norm,
                blend_alpha=args.masked26_blend_alpha,
                recon_zscore=bool(args.masked26_recon_zscore),
                keep_nonzero=bool(args.masked26_keep_nonzero),
                add_cellstats=bool(args.masked26_add_cellstats),
                cellstats_zscore=bool(args.masked26_cellstats_zscore),
                refine_k_labels=(len(np.unique(labels)) if bool(args.masked26_refine_k_labels) else None),
                config=cfg,
            )
        )
    if not outputs:
        raise RuntimeError("No masked26 ensemble outputs.")
    ref_shape = outputs[0].shape
    if any(out.shape != ref_shape for out in outputs):
        raise RuntimeError("Masked26 ensemble outputs have mismatched shapes.")
    if weights:
        if len(weights) != len(outputs):
            raise RuntimeError("masked26 ensemble weights length must match configs.")
        w = np.asarray(weights, dtype=np.float32)
        if not np.isfinite(w).all() or w.sum() <= 0:
            raise RuntimeError("Invalid masked26 ensemble weights.")
        w = w / w.sum()
        out = np.zeros_like(outputs[0], dtype=np.float32)
        for wi, arr in zip(w, outputs):
            out += wi * arr.astype(np.float32, copy=False)
        return out
    return np.mean(np.stack(outputs, axis=0), axis=0).astype(np.float32)


def _run_method(
    method: str,
    logcounts: np.ndarray,
    counts: Optional[np.ndarray],
    norm: Dict[str, np.ndarray],
    labels: np.ndarray,
    args: argparse.Namespace,
    seed: int,
    config_override: Optional[Dict[str, object]] = None,
) -> np.ndarray:

    method = method.lower()
    try:
        if method == "magic":
            return imp.run_magic(logcounts, n_jobs=args.n_jobs)
        if method == "autoclass":
            return imp.run_autoclass(logcounts, args.autoclass_dir, args.autoclass_kwargs)
        if method == "dca":
            if counts is None:
                raise RuntimeError("DCA requires raw counts assay.")
            size_factors = norm.get("size_factors")
            if size_factors is None:
                raise RuntimeError("Missing TrueCounts size factors for DCA normalization.")
            mu = imp.run_dca(
                counts=counts,
                dca_bin=args.dca_bin,
                ae_type=args.dca_type,
                epochs=args.dca_epochs,
                batch_size=args.dca_batch_size,
                threads=args.dca_threads,
                ridge=args.dca_ridge,
                verbose=args.verbose,
            )
            return imp.normalize_counts_to_logcounts(mu, size_factors)
        if method == "low_mse":
            bio_reg = 0.0
            if args.masked26_bio_reg_weight is not None:
                bio_reg = float(args.masked26_bio_reg_weight)
            if config_override is not None:
                    return _run_masked26_clustering(
                        logcounts,
                        counts,
                        bio_reg_weight=bio_reg,
                        seed=seed,
                        device=args.masked26_device,
                        output=args.masked26_output,
                        latent_norm=args.masked26_latent_norm,
                        blend_alpha=args.masked26_blend_alpha,
                        recon_zscore=bool(args.masked26_recon_zscore),
                        keep_nonzero=bool(args.masked26_keep_nonzero),
                        add_cellstats=bool(args.masked26_add_cellstats),
                        cellstats_zscore=bool(args.masked26_cellstats_zscore),
                        refine_k_labels=(len(np.unique(labels)) if bool(args.masked26_refine_k_labels) else None),
                        config=config_override,
                    )
            ensemble = getattr(args, "masked26_ensemble_configs_data", []) or []
            if ensemble:
                return _run_masked26_ensemble(
                    logcounts,
                    counts,
                    bio_reg=bio_reg,
                    seed=seed,
                    args=args,
                    labels=labels,
                    config_list=ensemble,
                    weights=getattr(args, "masked26_ensemble_weights_data", None),
                )
            return _run_masked26_clustering(
                logcounts,
                counts,
                bio_reg_weight=bio_reg,
                seed=seed,
                device=args.masked26_device,
                output=args.masked26_output,
                latent_norm=args.masked26_latent_norm,
                blend_alpha=args.masked26_blend_alpha,
                recon_zscore=bool(args.masked26_recon_zscore),
                keep_nonzero=bool(args.masked26_keep_nonzero),
                add_cellstats=bool(args.masked26_add_cellstats),
                cellstats_zscore=bool(args.masked26_cellstats_zscore),
                refine_k_labels=(len(np.unique(labels)) if bool(args.masked26_refine_k_labels) else None),
                config=getattr(args, "masked26_config_data", None),
            )
        if method == "balanced_mse":
            bio_reg = 1.0
            if args.masked26_bio_reg_weight is not None:
                bio_reg = float(args.masked26_bio_reg_weight)
            if config_override is not None:
                return _run_masked26_clustering(
                    logcounts,
                    counts,
                    bio_reg_weight=bio_reg,
                    seed=seed,
                    device=args.masked26_device,
                    output=args.masked26_output,
                    latent_norm=args.masked26_latent_norm,
                    blend_alpha=args.masked26_blend_alpha,
                    recon_zscore=bool(args.masked26_recon_zscore),
                    keep_nonzero=bool(args.masked26_keep_nonzero),
                    add_cellstats=bool(args.masked26_add_cellstats),
                    cellstats_zscore=bool(args.masked26_cellstats_zscore),
                    refine_k_labels=(len(np.unique(labels)) if bool(args.masked26_refine_k_labels) else None),
                    config=config_override,
                )
            ensemble = getattr(args, "masked26_ensemble_configs_data", []) or []
            if ensemble:
                return _run_masked26_ensemble(
                    logcounts,
                    counts,
                    bio_reg=bio_reg,
                    seed=seed,
                    args=args,
                    labels=labels,
                    config_list=ensemble,
                    weights=getattr(args, "masked26_ensemble_weights_data", None),
                )
            return _run_masked26_clustering(
                logcounts,
                counts,
                bio_reg_weight=bio_reg,
                seed=seed,
                device=args.masked26_device,
                output=args.masked26_output,
                latent_norm=args.masked26_latent_norm,
                blend_alpha=args.masked26_blend_alpha,
                recon_zscore=bool(args.masked26_recon_zscore),
                keep_nonzero=bool(args.masked26_keep_nonzero),
                add_cellstats=bool(args.masked26_add_cellstats),
                cellstats_zscore=bool(args.masked26_cellstats_zscore),
                refine_k_labels=(len(np.unique(labels)) if bool(args.masked26_refine_k_labels) else None),
                config=getattr(args, "masked26_config_data", None),
            )
        if method == "experiment":
            try:
                import experiment as exp
            except Exception as exc:
                raise RuntimeError(f"Failed to import experiment.py: {exc}") from exc
            exp_kwargs = dict(getattr(args, "experiment_kwargs", {}) or {})
            return exp.run_experiment_imputation(logcounts, seed=seed, **exp_kwargs)
    except BaseException as exc:
        raise RuntimeError(str(exc)) from exc
    raise ValueError(f"Unknown method: {method}")


def collect_rds_files(input_path: str) -> List[Path]:
    path = Path(input_path)
    if path.is_dir():
        return sorted(path.rglob("*.rds"))
    if path.is_file():
        return [path]
    raise FileNotFoundError(f"Input path not found: {input_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Clustering evaluation for SCE .rds files")
    parser.add_argument("input_path", help="Input .rds file or directory")
    parser.add_argument("output_dir", help="Output directory for clustering results")
    parser.add_argument(
        "--methods",
        default=None,
        help="Comma-separated list (magic,dca,autoclass,low_mse,balanced_mse,experiment) or 'all'.",
    )
    parser.add_argument(
        "methods_arg",
        nargs="?",
        default=None,
        help="Optional methods list (magic,dca,autoclass,low_mse,balanced_mse,experiment) or 'all'.",
    )
    parser.add_argument("--n-jobs", type=int, default=1, help="MAGIC n_jobs value")
    parser.add_argument("--n-repeat", type=int, default=5, help="Number of repeats per method")
    parser.add_argument(
        "--masked26-config",
        default=None,
        help="Path to JSON config for masked_imputation26 (applied to low_mse/balanced_mse).",
    )
    parser.add_argument(
        "--masked26-ensemble-configs",
        default="",
        help="Comma-separated list of JSON configs to ensemble for masked26 outputs (optional).",
    )
    parser.add_argument(
        "--masked26-ensemble-weights",
        default="",
        help="Comma-separated weights for masked26 ensemble configs (optional).",
    )
    parser.add_argument(
        "--masked26-refine-k-labels",
        action="store_true",
        help="When cluster_refine is enabled, use label count as cluster_refine_k (supervised tuning).",
    )
    parser.add_argument(
        "--masked26-dataset-map",
        default=None,
        help="JSON mapping of dataset name to masked26 overrides (method/config/output/cluster/etc).",
    )
    parser.add_argument(
        "--masked26-magic",
        action="store_true",
        help="Apply MAGIC to masked26 outputs before clustering.",
    )
    parser.add_argument(
        "--masked26-magic-blend",
        type=float,
        default=0.0,
        help="Blend MAGIC output with masked26 output (0 disables, 1 = all MAGIC).",
    )
    parser.add_argument(
        "--masked26-magic-source",
        default="recon",
        choices=["recon", "raw"],
        help="Source matrix for MAGIC when blending (recon=masked26 output, raw=original logcounts).",
    )
    parser.add_argument(
        "--masked26-magic-kwargs",
        default="",
        help="Comma-separated key=value args passed to MAGIC for masked26 outputs.",
    )
    parser.add_argument(
        "--masked26-device",
        default="cuda",
        help="Device for masked_imputation26 (cuda, cpu, auto).",
    )
    parser.add_argument(
        "--masked26-output",
        default="latent",
        help="masked_imputation26 output for clustering (recon, latent, recon+latent).",
    )
    parser.add_argument(
        "--masked26-add-cellstats",
        action="store_true",
        help="Append cell-level stats (log library size, zero fraction) to masked26 outputs.",
    )
    parser.add_argument(
        "--masked26-cellstats-zscore",
        action="store_true",
        help="Z-score cell stats before appending to masked26 outputs.",
    )
    parser.add_argument(
        "--masked26-select",
        default="single",
        choices=["single", "auto"],
        help="Selection mode for masked26 outputs (single uses masked26-output, auto picks best ASW among recon/latent/recon+latent).",
    )
    parser.add_argument(
        "--masked26-latent-norm",
        default="l2",
        help="Normalization for latent output (none, l2, zscore, center).",
    )
    parser.add_argument(
        "--masked26-blend-alpha",
        type=float,
        default=1.0,
        help="Blend recon with original logcounts for clustering (0..1).",
    )
    parser.add_argument(
        "--masked26-recon-zscore",
        action="store_true",
        help="Z-score recon features per gene before clustering.",
    )
    parser.add_argument(
        "--masked26-keep-nonzero",
        action="store_true",
        help="Preserve observed nonzeros after masked26 reconstruction (clustering only).",
    )
    parser.add_argument(
        "--masked26-pca-components",
        type=int,
        default=50,
        help="PCA components for masked26 methods (default 50).",
    )
    parser.add_argument(
        "--masked26-bio-reg-weight",
        type=float,
        default=None,
        help="Override bio_reg_weight for masked26 methods.",
    )
    parser.add_argument(
        "--masked26-kmeans-init",
        type=int,
        default=10,
        help="k-means n_init for masked26 methods (default 10).",
    )
    parser.add_argument(
        "--masked26-kmeans-iter",
        type=int,
        default=100,
        help="k-means max_iter for masked26 methods (default 100).",
    )
    parser.add_argument(
        "--masked26-smooth-k",
        type=int,
        default=0,
        help="Apply kNN smoothing (k neighbors) to masked26 outputs before clustering (0 disables).",
    )
    parser.add_argument(
        "--masked26-smooth-steps",
        type=int,
        default=0,
        help="Number of kNN smoothing steps for masked26 outputs.",
    )
    parser.add_argument(
        "--masked26-smooth-pca",
        type=int,
        default=50,
        help="PCA components used to build kNN graph for smoothing.",
    )
    parser.add_argument(
        "--masked26-cluster",
        default="kmeans",
        choices=["kmeans", "spectral", "gmm", "agglomerative", "auto"],
        help="Clustering algorithm for masked26 methods.",
    )
    parser.add_argument(
        "--masked26-auto-metric",
        default="asw",
        choices=["asw", "ch", "db"],
        help="Metric used to select best clustering when masked26-cluster=auto.",
    )
    parser.add_argument(
        "--masked26-embed",
        default="pca",
        choices=["pca", "tsne"],
        help="Embedding used before clustering for masked26 methods.",
    )
    parser.add_argument(
        "--masked26-top-genes",
        type=int,
        default=0,
        help="Use top-N highest-variance features for masked26 clustering (0 disables).",
    )
    parser.add_argument(
        "--masked26-spectral-k",
        type=int,
        default=15,
        help="Spectral clustering n_neighbors.",
    )
    parser.add_argument(
        "--masked26-gmm-cov",
        default="diag",
        choices=["full", "tied", "diag", "spherical"],
        help="GMM covariance_type for masked26 clustering.",
    )
    parser.add_argument(
        "--masked26-gmm-reg",
        type=float,
        default=1e-6,
        help="GMM reg_covar for masked26 clustering (default 1e-6).",
    )
    parser.add_argument(
        "--masked26-agg-linkage",
        default="ward",
        choices=["ward", "average", "complete"],
        help="Agglomerative linkage for masked26 clustering.",
    )
    parser.add_argument(
        "--masked26-tsne-dim",
        type=int,
        default=2,
        help="t-SNE embedding dimension for masked26 clustering.",
    )
    parser.add_argument(
        "--masked26-tsne-perplexity",
        type=float,
        default=5.0,
        help="t-SNE perplexity for masked26 clustering.",
    )
    parser.add_argument(
        "--masked26-tsne-iter",
        type=int,
        default=1000,
        help="t-SNE iterations for masked26 clustering.",
    )

    g_dca = parser.add_argument_group("DCA Options")
    g_dca.add_argument("--dca-bin", default="~/miniconda3/envs/dca_env/bin/dca", help="Path to DCA binary (for DCA method)")
    g_dca.add_argument("--dca-type", default=None, help="DCA --type (e.g., nb-conddisp)")
    g_dca.add_argument("--dca-epochs", type=int, default=None, help="DCA --epochs")
    g_dca.add_argument("--dca-batch-size", type=int, default=None, help="DCA --batch-size")
    g_dca.add_argument("--dca-threads", type=int, default=None, help="DCA --threads")
    g_dca.add_argument("--dca-ridge", type=float, default=None, help="DCA --ridge")

    g_ac = parser.add_argument_group("AutoClass Options")
    g_ac.add_argument("--autoclass-dir", default="AutoClass", help="Path to AutoClass repo (optional)")
    g_ac.add_argument("--autoclass-kwargs", default="", help="Comma-separated key=value overrides")

    g_exp = parser.add_argument_group("Experiment Options")
    g_exp.add_argument(
        "--experiment-kwargs",
        default="",
        help="Comma-separated key=value overrides for experiment method (e.g., epochs=50,bottleneck=16,masked_denoise=true).",
    )

    parser.add_argument("--verbose", action="store_true", help="Verbose method output (e.g., DCA)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = collect_rds_files(args.input_path)
    if not files:
        raise SystemExit("No .rds files found.")

    methods = parse_methods(args.methods_arg or args.methods)
    try:
        args.autoclass_kwargs = imp._parse_kv_pairs(args.autoclass_kwargs)
    except Exception as exc:
        raise SystemExit(f"Invalid --autoclass-kwargs: {exc}") from exc
    try:
        args.experiment_kwargs = imp._parse_kv_pairs(args.experiment_kwargs)
    except Exception as exc:
        raise SystemExit(f"Invalid --experiment-kwargs: {exc}") from exc
    try:
        args.masked26_magic_kwargs = imp._parse_kv_pairs(args.masked26_magic_kwargs)
    except Exception as exc:
        raise SystemExit(f"Invalid --masked26-magic-kwargs: {exc}") from exc
    masked26_config = imp._load_masked26_config(args.masked26_config)
    args.masked26_config_data = masked26_config
    ensemble_configs: List[Dict[str, object]] = []
    if args.masked26_ensemble_configs:
        paths = [p.strip() for p in str(args.masked26_ensemble_configs).split(",") if p.strip()]
        for p in paths:
            ensemble_configs.append(imp._load_masked26_config(p))
    args.masked26_ensemble_configs_data = ensemble_configs
    ensemble_weights = None
    if args.masked26_ensemble_weights:
        weights = [w.strip() for w in str(args.masked26_ensemble_weights).split(",") if w.strip()]
        try:
            ensemble_weights = [float(w) for w in weights]
        except ValueError as exc:
            raise SystemExit(f"Invalid masked26-ensemble-weights: {exc}") from exc
    args.masked26_ensemble_weights_data = ensemble_weights
    dataset_map = {}
    if args.masked26_dataset_map:
        dataset_map = json.loads(Path(args.masked26_dataset_map).read_text())
        if not isinstance(dataset_map, dict):
            raise SystemExit("masked26-dataset-map must be a JSON object.")

    method_rows: Dict[str, List[Dict[str, object]]] = {m: [] for m in methods}

    for rds_path in files:
        name = rds_path.stem
        print(f"\n--- Dataset: {name} ---")
        try:
            logcounts, counts, norm, labels, label_source = load_dataset(str(rds_path))
        except Exception as exc:
            print(f"  [ERROR] {name}: {exc}")
            continue

        override = dataset_map.get(name, {}) if dataset_map else {}

        for method in methods:
            print(f"  -> {method}")
            metrics_list: List[Dict[str, float]] = []
            runtimes: List[float] = []
            err_msg: Optional[str] = None
            for rep in range(args.n_repeat):
                seed = 42 + rep
                t0 = time.time()
                try:
                    method_run = method
                    config_override = None
                    local_args = args
                    if override and method in ("low_mse", "balanced_mse"):
                        method_run = override.get("method", method_run)
                        if override.get("config"):
                            config_override = imp._load_masked26_config(str(override["config"]))
                        # clone args to avoid mutating shared config
                        import copy as _copy
                        local_args = _copy.copy(args)
                        key_map = {
                            "output": "masked26_output",
                            "latent_norm": "masked26_latent_norm",
                            "cluster": "masked26_cluster",
                            "embed": "masked26_embed",
                            "pca_components": "masked26_pca_components",
                            "kmeans_init": "masked26_kmeans_init",
                            "kmeans_iter": "masked26_kmeans_iter",
                            "spectral_k": "masked26_spectral_k",
                            "gmm_cov": "masked26_gmm_cov",
                            "gmm_reg": "masked26_gmm_reg",
                            "agg_linkage": "masked26_agg_linkage",
                            "tsne_dim": "masked26_tsne_dim",
                            "tsne_perplexity": "masked26_tsne_perplexity",
                            "tsne_iter": "masked26_tsne_iter",
                            "recon_zscore": "masked26_recon_zscore",
                            "blend_alpha": "masked26_blend_alpha",
                            "smooth_k": "masked26_smooth_k",
                            "smooth_steps": "masked26_smooth_steps",
                            "smooth_pca": "masked26_smooth_pca",
                            "top_genes": "masked26_top_genes",
                            "keep_nonzero": "masked26_keep_nonzero",
                            "bio_reg_weight": "masked26_bio_reg_weight",
                            "select": "masked26_select",
                            "device": "masked26_device",
                        }
                        for key, attr in key_map.items():
                            if key in override:
                                setattr(local_args, attr, override[key])

                    log_imp = _run_method(method_run, logcounts, counts, norm, labels, local_args, seed, config_override=config_override)
                    if method in ("low_mse", "balanced_mse"):
                        output_mode = str(local_args.masked26_output or "recon").strip().lower()
                        latent_dim = None
                        if output_mode == "recon+latent":
                            latent_dim = _latent_dim_from_config(config_override or local_args.masked26_config_data)
                        recon_only = None
                        latent_only = None
                        if output_mode == "recon+latent":
                            if latent_dim and latent_dim > 0 and log_imp.shape[1] >= latent_dim:
                                recon_only = log_imp[:, :-latent_dim]
                                latent_only = log_imp[:, -latent_dim:]
                            else:
                                n_genes = logcounts.shape[1]
                                recon_only = log_imp[:, :n_genes]
                                latent_only = log_imp[:, n_genes:]
                        if args.masked26_smooth_k > 0 and args.masked26_smooth_steps > 0:
                            if recon_only is not None:
                                recon_only = _knn_smooth(
                                    recon_only,
                                    k=int(local_args.masked26_smooth_k),
                                    steps=int(local_args.masked26_smooth_steps),
                                    pca_components=int(local_args.masked26_smooth_pca),
                                )
                            else:
                                log_imp = _knn_smooth(
                                    log_imp,
                                    k=int(local_args.masked26_smooth_k),
                                    steps=int(local_args.masked26_smooth_steps),
                                    pca_components=int(local_args.masked26_smooth_pca),
                                )
                        if local_args.masked26_top_genes and local_args.masked26_top_genes > 0:
                            if recon_only is not None:
                                top_n = min(int(local_args.masked26_top_genes), recon_only.shape[1])
                                if top_n < recon_only.shape[1]:
                                    var = recon_only.var(axis=0)
                                    idx = np.argsort(var)[-top_n:]
                                    recon_only = recon_only[:, idx]
                            else:
                                top_n = min(int(local_args.masked26_top_genes), log_imp.shape[1])
                                if top_n < log_imp.shape[1]:
                                    var = log_imp.var(axis=0)
                                    idx = np.argsort(var)[-top_n:]
                                    log_imp = log_imp[:, idx]
                        if recon_only is not None:
                            log_imp = np.concatenate([recon_only, latent_only], axis=1)
                        if local_args.masked26_magic:
                            magic_kwargs = dict(getattr(local_args, "masked26_magic_kwargs", {}) or {})
                            if "n_jobs" not in magic_kwargs and hasattr(local_args, "n_jobs"):
                                magic_kwargs["n_jobs"] = int(local_args.n_jobs)
                            try:
                                import magic
                            except Exception as exc:
                                raise RuntimeError(f"Failed to import magic for masked26 postprocessing: {exc}") from exc
                            op = magic.MAGIC(**magic_kwargs)
                            log_imp = np.asarray(op.fit_transform(log_imp), dtype=np.float32)
                        if local_args.masked26_magic_blend and local_args.masked26_magic_blend > 0.0:
                            alpha = float(local_args.masked26_magic_blend)
                            if not (0.0 <= alpha <= 1.0):
                                raise RuntimeError("masked26-magic-blend must be in [0,1].")
                            magic_kwargs = dict(getattr(local_args, "masked26_magic_kwargs", {}) or {})
                            if "n_jobs" not in magic_kwargs and hasattr(local_args, "n_jobs"):
                                magic_kwargs["n_jobs"] = int(local_args.n_jobs)
                            try:
                                import magic
                            except Exception as exc:
                                raise RuntimeError(f"Failed to import magic for masked26 blending: {exc}") from exc
                            op = magic.MAGIC(**magic_kwargs)
                            source = logcounts if local_args.masked26_magic_source == "raw" else log_imp
                            magic_out = np.asarray(op.fit_transform(source), dtype=np.float32)
                            log_imp = (1.0 - alpha) * log_imp + alpha * magic_out
                        if local_args.masked26_select == "auto":
                            if str(local_args.masked26_output).strip().lower() != "recon+latent":
                                raise RuntimeError("masked26-select=auto requires masked26-output=recon+latent.")
                            if latent_dim and latent_dim > 0 and log_imp.shape[1] >= latent_dim:
                                recon = log_imp[:, :-latent_dim]
                                latent = log_imp[:, -latent_dim:]
                            else:
                                n_genes = logcounts.shape[1]
                                recon = log_imp[:, :n_genes]
                                latent = log_imp[:, n_genes:]
                            candidates = {"recon": recon, "latent": latent, "recon+latent": log_imp}
                            best = None
                            best_asw = -1e9
                            for _, mat in candidates.items():
                                if local_args.masked26_cluster == "auto":
                                    best_c = None
                                    best_asw_c = -1e9
                                    for cluster_name in ("kmeans", "spectral", "gmm", "agglomerative"):
                                        try:
                                            metrics_c = _cluster_labels(
                                                mat,
                                                labels,
                                                cluster=cluster_name,
                                                embed=str(local_args.masked26_embed),
                                                pca_components=int(local_args.masked26_pca_components),
                                                tsne_dim=int(local_args.masked26_tsne_dim),
                                                tsne_perplexity=float(local_args.masked26_tsne_perplexity),
                                                tsne_iter=int(local_args.masked26_tsne_iter),
                                                kmeans_init=int(local_args.masked26_kmeans_init),
                                                kmeans_iter=int(local_args.masked26_kmeans_iter),
                                                spectral_k=int(local_args.masked26_spectral_k),
                                                gmm_cov=str(local_args.masked26_gmm_cov),
                                                gmm_reg=float(local_args.masked26_gmm_reg),
                                                agg_linkage=str(local_args.masked26_agg_linkage),
                                                seed=seed,
                                            )
                                        except Exception:
                                            continue
                                        metric_mode = str(local_args.masked26_auto_metric).strip().lower()
                                        if metric_mode == "ch":
                                            score_c = float(metrics_c.get("CH", float("nan")))
                                        elif metric_mode == "db":
                                            score_c = -float(metrics_c.get("DB", float("nan")))
                                        else:
                                            score_c = float(metrics_c.get("ASW", float("nan")))
                                        if not np.isfinite(score_c):
                                            score_c = -1e9
                                        if score_c > best_asw_c:
                                            best_asw_c = score_c
                                            best_c = metrics_c
                                    metrics = best_c
                                else:
                                    metrics = _cluster_labels(
                                        mat,
                                        labels,
                                        cluster=str(local_args.masked26_cluster),
                                        embed=str(local_args.masked26_embed),
                                        pca_components=int(local_args.masked26_pca_components),
                                        tsne_dim=int(local_args.masked26_tsne_dim),
                                        tsne_perplexity=float(local_args.masked26_tsne_perplexity),
                                        tsne_iter=int(local_args.masked26_tsne_iter),
                                        kmeans_init=int(local_args.masked26_kmeans_init),
                                        kmeans_iter=int(local_args.masked26_kmeans_iter),
                                        spectral_k=int(local_args.masked26_spectral_k),
                                        gmm_cov=str(local_args.masked26_gmm_cov),
                                        gmm_reg=float(local_args.masked26_gmm_reg),
                                        agg_linkage=str(local_args.masked26_agg_linkage),
                                        seed=seed,
                                    )
                                metric_mode = str(local_args.masked26_auto_metric).strip().lower()
                                if metric_mode == "ch":
                                    score = float(metrics.get("CH", float("nan")))
                                elif metric_mode == "db":
                                    score = -float(metrics.get("DB", float("nan")))
                                else:
                                    score = float(metrics.get("ASW", float("nan")))
                                if not np.isfinite(score):
                                    score = -1e9
                                if score > best_asw:
                                    best_asw = score
                                    best = metrics
                            res = best if best is not None else evaluate_clustering(
                                log_imp,
                                labels,
                                n_components=int(local_args.masked26_pca_components),
                                n_init=int(local_args.masked26_kmeans_init),
                                max_iter=int(local_args.masked26_kmeans_iter),
                                seed=seed,
                            )
                        else:
                            if local_args.masked26_cluster == "auto":
                                best = None
                                best_asw = -1e9
                                for cluster_name in ("kmeans", "spectral", "gmm", "agglomerative"):
                                    try:
                                        metrics = _cluster_labels(
                                            log_imp,
                                            labels,
                                            cluster=cluster_name,
                                            embed=str(local_args.masked26_embed),
                                            pca_components=int(local_args.masked26_pca_components),
                                            tsne_dim=int(local_args.masked26_tsne_dim),
                                            tsne_perplexity=float(local_args.masked26_tsne_perplexity),
                                            tsne_iter=int(local_args.masked26_tsne_iter),
                                            kmeans_init=int(local_args.masked26_kmeans_init),
                                            kmeans_iter=int(local_args.masked26_kmeans_iter),
                                            spectral_k=int(local_args.masked26_spectral_k),
                                            gmm_cov=str(local_args.masked26_gmm_cov),
                                            gmm_reg=float(local_args.masked26_gmm_reg),
                                            agg_linkage=str(local_args.masked26_agg_linkage),
                                            seed=seed,
                                        )
                                    except Exception:
                                        continue
                                    metric_mode = str(local_args.masked26_auto_metric).strip().lower()
                                    if metric_mode == "ch":
                                        score = float(metrics.get("CH", float("nan")))
                                    elif metric_mode == "db":
                                        score = -float(metrics.get("DB", float("nan")))
                                    else:
                                        score = float(metrics.get("ASW", float("nan")))
                                    if not np.isfinite(score):
                                        score = -1e9
                                    if score > best_asw:
                                        best_asw = score
                                        best = metrics
                                res = best
                            else:
                                res = _cluster_labels(
                                    log_imp,
                                    labels,
                                    cluster=str(local_args.masked26_cluster),
                                    embed=str(local_args.masked26_embed),
                                    pca_components=int(local_args.masked26_pca_components),
                                    tsne_dim=int(local_args.masked26_tsne_dim),
                                    tsne_perplexity=float(local_args.masked26_tsne_perplexity),
                                    tsne_iter=int(local_args.masked26_tsne_iter),
                                    kmeans_init=int(local_args.masked26_kmeans_init),
                                    kmeans_iter=int(local_args.masked26_kmeans_iter),
                                    spectral_k=int(local_args.masked26_spectral_k),
                                        gmm_cov=str(local_args.masked26_gmm_cov),
                                        gmm_reg=float(local_args.masked26_gmm_reg),
                                        agg_linkage=str(local_args.masked26_agg_linkage),
                                        seed=seed,
                                    )
                    else:
                        res = evaluate_clustering(log_imp, labels, seed=seed)
                    metrics_list.append(res)
                except Exception as exc:
                    err_msg = str(exc)
                    print(f"    [ERROR] {method}: {exc}")
                    break
                runtimes.append(time.time() - t0)
            if metrics_list:
                def _mean_std(key: str) -> Tuple[float, float]:
                    vals = [m[key] for m in metrics_list]
                    return float(np.nanmean(vals)), float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0

                asw_m, asw_s = _mean_std("ASW")
                ari_m, ari_s = _mean_std("ARI")
                nmi_m, nmi_s = _mean_std("NMI")
                ps_m, ps_s = _mean_std("PS")
                rt_m = float(np.mean(runtimes)) if runtimes else float("nan")
                rt_s = float(np.std(runtimes, ddof=1)) if len(runtimes) > 1 else 0.0
                row = {
                    "dataset": name,
                    "ASW": asw_m,
                    "ASW_std": asw_s,
                    "ARI": ari_m,
                    "ARI_std": ari_s,
                    "NMI": nmi_m,
                    "NMI_std": nmi_s,
                    "PS": ps_m,
                    "PS_std": ps_s,
                    "runtime_sec": rt_m,
                    "runtime_sec_std": rt_s,
                    "n_repeats": len(metrics_list),
                    "n_cells": int(logcounts.shape[0]),
                    "n_genes": int(logcounts.shape[1]),
                    "label_source": label_source,
                    "error": err_msg or "",
                }
            else:
                row = {
                    "dataset": name,
                    "ASW": float("nan"),
                    "ASW_std": float("nan"),
                    "ARI": float("nan"),
                    "ARI_std": float("nan"),
                    "NMI": float("nan"),
                    "NMI_std": float("nan"),
                    "PS": float("nan"),
                    "PS_std": float("nan"),
                    "runtime_sec": float("nan"),
                    "runtime_sec_std": float("nan"),
                    "n_repeats": 0,
                    "n_cells": int(logcounts.shape[0]),
                    "n_genes": int(logcounts.shape[1]),
                    "label_source": label_source,
                    "error": err_msg or "method failed",
                }
            method_rows[method].append(row)
            if metrics_list:
                print(
                    f"    ASW={row['ASW']:.4f} ARI={row['ARI']:.4f} NMI={row['NMI']:.4f} PS={row['PS']:.4f}"
                )

    # save per-method tables
    import csv

    columns = [
        "dataset",
        "ASW",
        "ASW_std",
        "ARI",
        "ARI_std",
        "NMI",
        "NMI_std",
        "PS",
        "PS_std",
        "runtime_sec",
        "runtime_sec_std",
        "n_repeats",
        "n_cells",
        "n_genes",
        "label_source",
        "error",
    ]
    for method, rows in method_rows.items():
        out_path = out_dir / f"{method}_clustering_table.tsv"
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns, delimiter="\t")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        print(f"Saved {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
