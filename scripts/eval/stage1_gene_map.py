#!/usr/bin/env python3
"""Stage-1 HEST spatial gene-map evaluation.

Frozen Stage-1 tx encoder -> spot embedding -> Ridge probe -> selected gene
expression maps.  This mirrors stage15_gene_map.py but uses Stage-1
representations (h_tx/chunk_state/spot_state) instead of the spatial encoder.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from mm_align.data.gene_norm import GeneNormalizer
from mm_align.evaluation.stage1_benchmarks import chunk_view_embeddings_from_encoder, _spearman
from mm_align.evaluation.gene_imputation_metrics import (
    pearson_1d as _pearson,
    rmse_zscore as _rmse_zscore,
    ssim_1d as _ssim_1d,
    jsd_1d as _jsd_1d,
)
from mm_align.utils import get_logger

# Reuse plotting/gene-selection helpers from Stage1.5 evaluator so the two
# stages produce comparable artifacts and metric files.
from stage15_gene_map import (  # type: ignore
    _load_stage1_tx,
    _embed_tx,
    _load_gene_names,
    _resolve_shard,
    _split_paths,
    _load_shard_arrays,
    _auto_select_spatial_genes,
    _plot_gene_map,
    _plot_scc_barplot,
    _write_hest_style_summary,
)

log = get_logger("eval_stage1_gene_map")


def _parse_reps(text: str) -> list[str]:
    if text.strip().lower() == "all":
        return ["h_tx", "chunk_state", "spot_state"]
    valid = {"h_tx", "chunk_state", "spot_state"}
    reps = [x.strip() for x in text.split(",") if x.strip()]
    bad = sorted(set(reps) - valid)
    if bad:
        raise SystemExit(f"unknown representations={bad}; use h_tx,chunk_state,spot_state,all")
    return reps or ["spot_state"]


@torch.no_grad()
def _encode_representations(tx_encoder, hvg_norm: np.ndarray, reps: list[str], *,
                            device: str, batch: int, chunk_n: int,
                            chunk_len: int, seed: int) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    if "h_tx" in reps:
        out["h_tx"] = _embed_tx(tx_encoder, hvg_norm, device=device, batch=batch)
    if "chunk_state" in reps or "spot_state" in reps:
        views = chunk_view_embeddings_from_encoder(
            tx_encoder, hvg_norm,
            n_chunks=chunk_n,
            chunk_len=min(chunk_len, hvg_norm.shape[1]),
            dynamic=True,
            batch_size=max(1, min(batch, 128)),
            max_spots=max(1, hvg_norm.shape[0]),
            seed=seed,
            device=device,
        )
        if "chunk_state" in reps:
            out["chunk_state"] = views["z_chunk"].astype(np.float32)
        if "spot_state" in reps:
            out["spot_state"] = views["z_spot"].astype(np.float32)
    return out


def _load_encoded_sample(path: Path, tx_encoder, normalizer: GeneNormalizer,
                         vocab_keep: np.ndarray | None, reps: list[str], *,
                         device: str, batch: int, chunk_n: int,
                         chunk_len: int, seed: int) -> dict:
    _full, hvg_eff, coords, _uni = _load_shard_arrays(path, vocab_keep)
    hvg_norm = normalizer.apply_np(hvg_eff).astype(np.float32, copy=False)
    emb = _encode_representations(
        tx_encoder, hvg_norm, reps,
        device=device, batch=batch, chunk_n=chunk_n, chunk_len=chunk_len, seed=seed,
    )
    return {
        "sample_id": path.stem.split(".")[0],
        "emb": emb,
        "hvg_eff": hvg_eff.astype(np.float32, copy=False),
        "coords": coords.astype(np.float32, copy=False),
    }


def _copy_top_gene_maps(df: pd.DataFrame, out_dir: Path, *, top_k: int = 5) -> None:
    """Collect the top-SCC gene-map figures into one easy-to-review folder."""
    if df.empty or "spearman_scc" not in df.columns:
        return
    import shutil
    top = df.sort_values("spearman_scc", ascending=False).head(int(top_k))
    dst = out_dir / "top_scc_gene_maps"
    dst.mkdir(parents=True, exist_ok=True)
    for rank, row in enumerate(top.itertuples(index=False), 1):
        rep = getattr(row, "representation")
        sample = getattr(row, "sample_id")
        gene = getattr(row, "gene")
        src = out_dir / rep / sample / f"{gene}.png"
        if src.exists():
            score = float(getattr(row, "spearman_scc"))
            shutil.copyfile(src, dst / f"rank{rank:02d}_scc{score:+.3f}_{rep}_{sample}_{gene}.png")
    top.to_csv(dst / "top_scc_gene_maps.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage1-ckpt", required=True)
    ap.add_argument("--prepared-dir", default="results/cache/prepared_expanded")
    ap.add_argument("--split", default="test", choices=("train", "val", "test"))
    ap.add_argument("--source", default="hest", choices=("hest", "all"))
    ap.add_argument("--samples", nargs="*", default=None)
    ap.add_argument("--representations", default="spot_state")
    ap.add_argument("--genes", nargs="*", default=[])
    ap.add_argument("--auto-select-genes", type=int, default=4)
    ap.add_argument("--probe-train-samples", type=int, default=20)
    ap.add_argument("--max-train-spots", type=int, default=20000)
    ap.add_argument("--tx-batch", type=int, default=256)
    ap.add_argument("--tx-pooling-mode", default="ckpt",
                    choices=("ckpt", "cls", "token_mean", "cls_token_mean_sum", "cls_token_mean_avg",
                             "cls_mean_sum", "cls_mean_avg", "mean"),
                    help="Override Stage1 tx_encoder spot readout at eval time.")
    ap.add_argument("--chunk-n", type=int, default=4)
    ap.add_argument("--chunk-len", type=int, default=256)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--no-viz", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    prepared = Path(args.prepared_dir)
    reps = _parse_reps(args.representations)

    tx_encoder, _tx_dim, vocab_keep, gene_norm_cfg = _load_stage1_tx(
        Path(args.stage1_ckpt), device, tx_pooling_mode=args.tx_pooling_mode
    )
    with h5py.File(_split_paths(prepared, "train", source="hest")[0], "r") as f0:
        full_dim = int(f0["hvg_log"].shape[1])
    hvg_dim = int(len(vocab_keep)) if vocab_keep is not None else full_dim
    normalizer = GeneNormalizer(gene_norm_cfg, full_hvg_dim=full_dim, hvg_dim=hvg_dim,
                                vocab_keep_indices=vocab_keep)
    genes_eff = _load_gene_names(prepared, vocab_keep)
    gene_to_eff = {g.upper(): i for i, g in enumerate(genes_eff)}

    if args.samples:
        eval_paths = [p for sid in args.samples if (p := _resolve_shard(prepared, sid)) is not None]
    else:
        eval_paths = _split_paths(prepared, args.split, source=args.source)[:5]
    if not eval_paths:
        raise SystemExit(f"No eval shards found for split={args.split}")

    requested_genes = list(args.genes or [])
    if args.auto_select_genes > 0:
        auto = _auto_select_spatial_genes(
            eval_paths, vocab_keep, genes_eff, int(args.auto_select_genes),
            exclude={g.upper() for g in requested_genes},
        )
        log.info(f"auto-selected Stage1 spatial genes: {auto}")
        requested_genes.extend(auto)
    seen = set()
    requested_genes = [g for g in requested_genes if not (g.upper() in seen or seen.add(g.upper()))]
    if not requested_genes:
        raise SystemExit("No genes requested. Use --genes ... or --auto-select-genes N")
    missing = [g for g in requested_genes if g.upper() not in gene_to_eff]
    if missing:
        raise SystemExit(f"Genes not in effective Stage1 vocab: {missing}")
    gene_idx = np.array([gene_to_eff[g.upper()] for g in requested_genes], dtype=np.int64)

    train_paths = _split_paths(prepared, "train", source="hest")[: args.probe_train_samples]
    if not train_paths:
        raise SystemExit("No HEST train shards found for probe training")
    rng = np.random.default_rng(0)
    per_train = max(1, args.max_train_spots // max(1, len(train_paths)))
    train_records = []
    log.info(f"encoding {len(train_paths)} train shards for Stage1 Ridge probe...")
    for i, p in enumerate(train_paths):
        rec = _load_encoded_sample(
            p, tx_encoder, normalizer, vocab_keep, reps,
            device=device, batch=args.tx_batch, chunk_n=args.chunk_n,
            chunk_len=args.chunk_len, seed=i,
        )
        n = rec["hvg_eff"].shape[0]
        sel = rng.choice(n, min(n, per_train), replace=False) if n > per_train else np.arange(n)
        rec["sel"] = sel
        train_records.append(rec)

    out_dir = Path(args.out_dir)
    rows = []
    all_spot_tables = []
    for rep in reps:
        Xtr = np.concatenate([r["emb"][rep][r["sel"]] for r in train_records], axis=0)
        Ytr = np.concatenate([r["hvg_eff"][r["sel"]][:, gene_idx] for r in train_records], axis=0)
        probe = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        probe.fit(Xtr, Ytr)

        for e_i, p in enumerate(eval_paths):
            rec = _load_encoded_sample(
                p, tx_encoder, normalizer, vocab_keep, reps,
                device=device, batch=args.tx_batch, chunk_n=args.chunk_n,
                chunk_len=args.chunk_len, seed=1000 + e_i,
            )
            pred = np.asarray(probe.predict(rec["emb"][rep]), dtype=np.float32)
            gt = rec["hvg_eff"][:, gene_idx]
            for j, gene in enumerate(requested_genes):
                scc = _spearman(pred[:, j], gt[:, j])
                pcc = _pearson(pred[:, j], gt[:, j])
                rmse = _rmse_zscore(gt[:, j], pred[:, j])
                ssim = _ssim_1d(gt[:, j], pred[:, j])
                jsd = _jsd_1d(gt[:, j], pred[:, j])
                rows.append({
                    "ckpt": Path(args.stage1_ckpt).parent.name,
                    "stage": "stage1",
                    "representation": rep,
                    "sample_id": rec["sample_id"],
                    "gene": gene,
                    "metric_unit": "per_gene_per_sample",
                    "pcc_definition": "Pearson across spots for one gene within one sample",
                    "scc_definition": "Spearman across spots for one gene within one sample",
                    "spearman_scc": scc,
                    "pearson": pcc,
                    "rmse_zscore": rmse,
                    "ssim": ssim,
                    "jsd": jsd,
                    "n_spots": int(gt.shape[0]),
                    "gt_nonzero_frac": float((gt[:, j] > 0).mean()),
                })
                sample_dir = out_dir / rep / rec["sample_id"]
                sample_dir.mkdir(parents=True, exist_ok=True)
                if not args.no_viz:
                    _plot_gene_map(
                        rec["coords"], gt[:, j], pred[:, j], rec["sample_id"], gene,
                        {"SCC": scc, "PCC": pcc, "SSIM": ssim, "JSD": jsd, "RMSE_z": rmse},
                        sample_dir / f"{gene}.png",
                    )
                spot_df = pd.DataFrame({
                    "sample_id": rec["sample_id"],
                    "spot_index": np.arange(gt.shape[0], dtype=np.int64),
                    "gene": gene,
                    "representation": rep,
                    "x": rec["coords"][:, 0],
                    "y": rec["coords"][:, 1],
                    "gt": gt[:, j],
                    "pred": pred[:, j],
                })
                spot_df.to_csv(sample_dir / f"{gene}_spots.csv", index=False)
                all_spot_tables.append(spot_df)

    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "gene_map_scc.csv", index=False)
    _write_hest_style_summary(all_spot_tables, out_dir)
    if not args.no_viz:
        _plot_scc_barplot(df, out_dir / "gene_map_scc_barplot.png")
        _copy_top_gene_maps(df, out_dir, top_k=5)
    log.info(f"wrote {out_dir / 'gene_map_scc.csv'}")
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
