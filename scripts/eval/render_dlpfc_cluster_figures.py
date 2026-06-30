#!/usr/bin/env python3
"""Regenerate DLPFC cluster summary figures from an existing per-sample CSV."""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
_spec = importlib.util.spec_from_file_location("_dlpfc_eval", Path(__file__).resolve().parent / "dlpfc_eval.py")
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-sample", required=True, help="DLPFC per-sample CSV from scripts/eval/dlpfc_eval.py")
    ap.add_argument("--out-dir", required=True, help="Figure root, e.g. results/figures/<prefix>")
    ap.add_argument("--methods", default="auto", help="Comma list or auto")
    args = ap.parse_args()

    df = pd.read_csv(args.per_sample)
    if args.methods == "auto":
        methods = []
        for c in df.columns:
            for suf in ("_ari", "_nmi", "_chaos", "_pas", "_asw_spatial"):
                if c.endswith(suf):
                    methods.append(c[:-len(suf)])
        methods = sorted(set(methods))
    else:
        methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    out_root = Path(args.out_dir)
    for (ck_name, rep), grp in df.groupby(["ckpt", "representation"]):
        out = out_root / "zero_shot" / "cluster" / "dlpfc" / str(ck_name) / str(rep) / "method_summary.png"
        _mod._plot_method_summary_bar(grp, methods, out)
        print(out)


if __name__ == "__main__":
    main()
