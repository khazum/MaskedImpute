#!/usr/bin/env python3
"""Evaluate post-hoc MaskImpute transforms on synthetic benchmark metrics."""
from __future__ import annotations
import argparse, csv, importlib.util, json, sys, time
from pathlib import Path
from typing import Dict, List
import numpy as np
import torch
from rds2py import read_rds

REPO_ROOT=Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0,str(REPO_ROOT))
import masked_imputation26 as mi26
import run_imputation as bench
_spec=importlib.util.spec_from_file_location('real_tune', REPO_ROOT/'scripts'/'tune_maskimpute_real_clustering.py')
real_tune=importlib.util.module_from_spec(_spec); sys.modules['real_tune']=real_tune; _spec.loader.exec_module(real_tune)

def collect(input_dir: Path, datasets: str) -> List[Path]:
    files=sorted(input_dir.rglob('*.rds'))
    if datasets:
        wanted=set(x.strip() for x in datasets.split(',') if x.strip())
        files=[p for p in files if bench.dataset_name_from_path(p) in wanted]
    if not files: raise FileNotFoundError(input_dir)
    return files


def _load_labels(path: Path, column: str) -> np.ndarray | None:
    try:
        sce = read_rds(str(path))
        values = bench._get_coldata_column(sce, column)
    except Exception:
        values = None
    if values is None:
        return None
    labels = np.asarray(values).reshape(-1)
    if labels.size < 2 or len(set(map(str, labels))) < 2:
        return None
    return np.array([str(x) for x in labels], dtype=object)


def _downstream_summary(rows: List[Dict[str, object]], n_labels: int) -> Dict[str, float]:
    import pandas as pd

    df = pd.DataFrame(rows)
    out = {
        "ari_grid_mean": float(df["ari"].mean()),
        "nmi_grid_mean": float(df["nmi"].mean()),
        "ari_grid_max": float(df["ari"].max()),
        "nmi_grid_max": float(df["nmi"].max()),
    }
    matched = []
    for _, sub in df.groupby("seed"):
        tmp = sub.copy()
        tmp["cluster_delta"] = (tmp["n_clusters"] - n_labels).abs()
        tmp["res_delta"] = (tmp["resolution"] - 1.0).abs()
        matched.append(tmp.sort_values(["cluster_delta", "res_delta", "n_pcs", "n_neighbors"]).iloc[0])
    m = pd.DataFrame(matched)
    out["ari_matched_mean"] = float(m["ari"].mean())
    out["nmi_matched_mean"] = float(m["nmi"].mean())
    out["n_clusters_matched_mean"] = float(m["n_clusters"].mean())
    return out

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input-dir',type=Path,default=Path('synthetic_datasets/simulated_data/test'))
    ap.add_argument('--out-dir',type=Path,default=Path('results_real_data/tuning_maskimpute/synthetic_transforms'))
    ap.add_argument('--datasets',default='')
    ap.add_argument('--transforms',default='raw,all_obs0.1,all_obs0.25,all_obs0.5,all_obs0.75')
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--config',default='{"name":"default"}')
    ap.add_argument('--downstream',action='store_true',help='Also compute Leiden ARI/NMI against synthetic labels.')
    ap.add_argument('--label-column',default='Group')
    ap.add_argument('--cluster-seeds',default='42')
    ap.add_argument('--n-pcs-grid',default='20,50')
    ap.add_argument('--n-neighbors-grid',default='10,15,30')
    ap.add_argument('--resolution-grid',default='0.2,0.5,1.0')
    args=ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    config=real_tune.load_json(args.config); real_tune.apply_config(config)
    transforms=[x.strip() for x in args.transforms.split(',') if x.strip()]
    cluster_seeds=real_tune.phase5.parse_int_list(args.cluster_seeds)
    n_pcs_values=real_tune.phase5.parse_int_list(args.n_pcs_grid)
    n_neighbors_values=real_tune.phase5.parse_int_list(args.n_neighbors_grid)
    resolutions=real_tune.phase5.parse_float_list(args.resolution_grid)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rows=[]
    downstream_rows=[]
    for path in collect(args.input_dir,args.datasets):
        ds=bench.dataset_name_from_path(path)
        print('===',ds, flush=True)
        dataset=bench.load_dataset(str(path))
        logcounts=np.asarray(dataset['logcounts'],dtype=np.float32)
        log_true=np.asarray(dataset['log_true'],dtype=np.float32)
        counts=dataset.get('counts')
        if counts is None: counts=np.clip(np.expm1(logcounts*np.log(2.0)),0,None).astype(np.float32)
        else: counts=np.clip(np.asarray(counts,dtype=np.float32),0,None)
        zeros=counts<=0; counts_max=counts.max(axis=0)
        masks=bench.compute_masks(log_true, logcounts, marker_gene_mask=np.asarray(dataset['marker_gene_mask'],dtype=bool))
        labels=_load_labels(path,args.label_column) if args.downstream else None
        p_bio=mi26.splat_cellaware_bio_prob(counts=counts, zeros_obs=zeros, disp_mode=mi26.BIO_PARAMS['disp_mode'], use_cell_factor=mi26.BIO_PARAMS['use_cell_factor'])
        mi26.set_seed(args.seed)
        t0=time.perf_counter()
        recon=mi26.train_autoencoder_reconstruct(logcounts=logcounts, counts_max=counts_max, p_bio=p_bio, device=device, fast_mode=True, amp_enabled=True, compile_enabled=False, fast_batch_mult=2, num_workers=2)
        p_ref=mi26.refine_bio_prob_with_reconstruction(recon_log=recon, counts_obs=counts, zeros_obs=zeros)
        imp=mi26.apply_zero_gate(recon, p_ref, zeros)
        runtime=time.perf_counter()-t0
        for tr in transforms:
            mat=real_tune.transform_matrix(tr, imp, logcounts, counts)
            met=bench.compute_error_metrics(mat, log_true, masks)
            row={'dataset':ds,'transform':tr,'seed':args.seed,'runtime_sec':runtime}
            for k in ['mse','mse_dropout','mse_biozero','mae','gnrmse']:
                row[k]=met[k]
            if labels is not None:
                grid_rows=[]
                for cseed in cluster_seeds:
                    grid_rows.extend(real_tune.phase5.cluster_grid(
                        mat,
                        labels,
                        None,
                        n_pcs_values=n_pcs_values,
                        n_neighbors_values=n_neighbors_values,
                        resolutions=resolutions,
                        seed=cseed,
                        silhouette_sample_size=5000,
                    ))
                ds_summary=_downstream_summary(grid_rows, n_labels=len(set(labels.tolist())))
                row.update(ds_summary)
                for grow in grid_rows:
                    grow.update({"dataset":ds,"transform":tr,"seed":args.seed,"cluster_seed":grow.get("seed")})
                    downstream_rows.append(grow)
            rows.append(row); print(json.dumps(row, sort_keys=True), flush=True)
    out=args.out_dir/'metrics.tsv'
    with out.open('w',newline='') as fh:
        w=csv.DictWriter(fh,delimiter='\t',fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    # summary by transform
    import pandas as pd
    df=pd.DataFrame(rows)
    metric_cols=['mse','mse_dropout','mse_biozero','mae','gnrmse']
    if args.downstream and 'ari_grid_mean' in df.columns:
        metric_cols += ['ari_grid_mean','nmi_grid_mean','ari_matched_mean','nmi_matched_mean']
    summary=df.groupby('transform')[metric_cols].mean().reset_index()
    summary.to_csv(args.out_dir/'summary.tsv',sep='\t',index=False)
    if downstream_rows:
        pd.DataFrame(downstream_rows).to_csv(args.out_dir/'downstream_grid.tsv',sep='\t',index=False)
    print(summary.to_string(index=False)); print('Wrote',out)
if __name__=='__main__': main()
