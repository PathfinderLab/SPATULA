"""Write stage-specific split manifests from prepared `splits.json`.

The global split stays the source of truth. This script emits explicit views so
experiments and reports can distinguish what each stage is supposed to train,
validate, and test on.

Usage:
    python scripts/data/write_stage_splits.py \
        --prepared-dir results/cache/prepared_expanded

Outputs:
    splits_stage1.json       RNA foundation: mixed-source train/val/test.
    splits_stage15.json      Spatial encoder: mixed-source train/val + DLPFC external test note.
    splits_stage2_hest.json  Alignment: HEST-only train/val/test.
    split_manifest.json      Human-readable task/eval intent.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


def _source_suffix(prepared: Path, sample_id: str) -> tuple[str, str] | None:
    for suffix, source in (("", "hest"), (".st1k", "st1k"), (".spatialcorpus", "spatialcorpus")):
        if (prepared / f"{sample_id}{suffix}.h5").exists():
            return suffix, source
    return None


def _filter_by_source(prepared: Path, ids: Iterable[str], source: str) -> list[str]:
    out = []
    for sid in ids:
        hit = _source_suffix(prepared, sid)
        if hit is not None and hit[1] == source:
            out.append(sid)
    return out


def _source_counts(prepared: Path, ids: Iterable[str]) -> dict[str, int]:
    counts = {"hest": 0, "st1k": 0, "spatialcorpus": 0, "missing": 0}
    for sid in ids:
        hit = _source_suffix(prepared, sid)
        counts[hit[1] if hit else "missing"] += 1
    return counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared-dir", default="results/cache/prepared_expanded")
    ap.add_argument("--dlpfc-dir", default="/data/spatiallibd")
    args = ap.parse_args()

    prepared = Path(args.prepared_dir)
    splits_path = prepared / "splits.json"
    if not splits_path.exists():
        raise SystemExit(f"{splits_path} not found")
    splits = json.loads(splits_path.read_text())

    stage1 = {
        "stage": "stage1",
        "purpose": "RNA/spot encoder foundation selection.",
        "train": splits.get("train", []),
        "val": splits.get("val", []),
        "test": splits.get("test", []),
        "eval_tasks": {
            "intrinsic": ["masked_symbol_topk", "embedding_effective_rank", "gene_set_monitor"],
            "linear_probe": ["frozen_h_tx_to_hvg_expression"],
            "extrinsic": ["dlpfc_layer_probe_as_external_check"],
        },
        "source_counts": {k: _source_counts(prepared, v) for k, v in splits.items()},
    }

    stage15 = {
        "stage": "stage1.5",
        "purpose": "Spatial context-aware encoder selection after a full Stage-1 checkpoint.",
        "requires": {"stage1_checkpoint": "Pass a trained ckpt_tx_encoder_best.pt to Stage-1.5."},
        "train": splits.get("train", []),
        "val": splits.get("val", []),
        "test": splits.get("test", []),
        "external_test": {"dlpfc_dir": args.dlpfc_dir},
        "eval_tasks": {
            "intrinsic": ["spatial_jepa_prediction", "embedding_health"],
            "linear_probe": ["dlpfc_layer_probe"],
            "extrinsic": ["spatial_clustering_ari_nmi", "knn_layer_purity"],
        },
        "source_counts": {k: _source_counts(prepared, v) for k, v in splits.items()},
    }

    stage2 = {
        "stage": "stage2",
        "purpose": "HEST image-transcriptomics alignment and downstream image tasks.",
        "train": _filter_by_source(prepared, splits.get("train", []), "hest"),
        "val": _filter_by_source(prepared, splits.get("val", []), "hest"),
        "test": _filter_by_source(prepared, splits.get("test", []), "hest"),
        "eval_tasks": {
            "zero_shot": ["cross_modal_retrieval", "alignment_uniformity", "modality_gap"],
            "linear_probe": ["image_to_hvg_expression", "slide_level_mil_when_labels_available"],
            "extrinsic": ["HEST_prediction_tasks", "SEAL_PathBench_style_image_tasks"],
        },
        "source_counts": {"hest_only": {k: len(v) for k, v in {
            "train": _filter_by_source(prepared, splits.get("train", []), "hest"),
            "val": _filter_by_source(prepared, splits.get("val", []), "hest"),
            "test": _filter_by_source(prepared, splits.get("test", []), "hest"),
        }.items()}},
    }

    manifest = {
        "source": str(splits_path),
        "note": "Stage manifests are views over the global split; they do not reshuffle samples.",
        "files": {
            "stage1": "splits_stage1.json",
            "stage1.5": "splits_stage15.json",
            "stage2": "splits_stage2_hest.json",
        },
    }

    outputs = {
        "splits_stage1.json": stage1,
        "splits_stage15.json": stage15,
        "splits_stage2_hest.json": stage2,
        "split_manifest.json": manifest,
    }
    for name, obj in outputs.items():
        path = prepared / name
        path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
