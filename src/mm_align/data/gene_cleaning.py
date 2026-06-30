"""Gene-symbol normalisation + noise-pattern filtering.

Used in two places:
    1. `scripts/data/prepare.py` — per-sample cleaning before vocab build / shard write
    2. `scripts/eval/validate_vocab.py` — re-projection audits

Why this lives in mm_align (not just scripts/): the rules embody the
project's biological priors (which patterns to drop, when to rescue via
GTF) and must stay identical between training-time prepare and audit-time
re-projection — otherwise validate_vocab fails.

The cleaning pipeline (in order):
    1. Upper-case + version-strip (TP53.4 → TP53)
    2. Genome prefix strip (GRCH38_TP53 → TP53)
    3. Suffix normalisation (TP53.AS1 → TP53-AS1)
    4. ENSG → HGNC via GTF map (when ENSG-like ids are present)
    5. Drop pattern-hit "noise" symbols (MT-, RPS, RPL, BLANK_, AMBIGUOUS[…], …)
    6. GTF rescue: keep protein_coding / IG_*/TR_* even if they pattern-match
       (TP53, MMP1, MAP2 look like pseudogenes but aren't)
    7. NEVER rescue MT-/RPS/RPL — biological house-keeping noise even if PC

Outputs:
    cleaned AnnData (same row count) + count of noise drops.
"""
from __future__ import annotations
import logging
import re

import numpy as np

log = logging.getLogger("gene_cleaning")


# ── Regex constants ─────────────────────────────────────────────────────────

_PREFIX_RE  = re.compile(r"^(?:GRCH38|HG38|SS11|CH38|GRCH37)+_+", re.IGNORECASE)
_VERSION_RE = re.compile(r"\.[0-9]+$")
_SUFFIX_RE  = re.compile(r"[._](AS|IT|OT)([0-9]+)$", re.IGNORECASE)
_ENSG_RE    = re.compile(r"^ENS[A-Z]*G\d+(\.\d+)?$", re.IGNORECASE)

# Symbols to drop — each is an HGNC-noise pattern observed in real samples.
_NOISE_PATTERNS = [
    r"^MT-", r"^MT\.",                                       # mitochondrial
    r"^RPS", r"^RPL",                                        # ribosomal
    r"^__",
    r"^BLANK_", r"^NEGCONTROL", r"^UNASSIGNED",
    r"^[0-9]",
    r"^A[CDFJLP][0-9]+\.[0-9]+", r"^A[CDFJLP][0-9]+",
    r"^LOC[0-9]+", r"^RP[0-9]+", r"^LINC[0-9]+",
    r"^.{3,}P[0-9]+$",                                       # X+pseudogene suffix
    r"^DEPRECATED",
    # STAR / CellRanger multi-mapper aggregator labels (~571 of the
    # prepared_4k "unknown" bucket were this single pattern).
    r"^AMBIGUOUS\[",
    # RefSeq NC_ contigs — SARS-CoV-2 ORF probes etc.
    r"^NC_[0-9]",
    # BAC / fosmid clone IDs surfacing as gene names in old GTFs.
    r"^CT[A-D]-[0-9]", r"^GS[12]-", r"^LA16C-", r"^KB-[0-9]",
    r"^CH[0-9]+-[0-9]", r"^LL[0-9A-Z]+-", r"^XX[A-Z]+",
]
_NOISE_RE = re.compile("|".join(_NOISE_PATTERNS), re.IGNORECASE)

# Symbols we NEVER rescue via the protein-coding guard.  MT-CO1 / RPL37A are
# technically protein_coding but biologically house-keeping / mitochondrial
# noise that we want gone regardless of GTF.
_NEVER_RESCUE_RE = re.compile(r"^MT[-.]|^RPS|^RPL", re.IGNORECASE)

# GTF gene_types that pass the rescue gate (anything that codes for a protein
# or rearranges into one).
_ALLOW_TYPES = {
    "protein_coding",
    "IG_C_gene", "IG_V_gene", "IG_J_gene", "IG_D_gene",
    "TR_C_gene", "TR_V_gene", "TR_J_gene", "TR_D_gene",
}


# ── Functions ───────────────────────────────────────────────────────────────

def clean_symbol(name: str, gtf_map: dict | None = None) -> str:
    """Normalise a single gene name.

    Strips genome prefix / version / suffix, upper-cases, and (when
    `gtf_map` is provided AND the input looks like an Ensembl ID) resolves
    to the HGNC symbol.  Unresolved Ensembl IDs are returned as-is — the
    caller decides whether to drop them.
    """
    if not isinstance(name, str):
        return ""
    s = name.upper().strip()
    if gtf_map is not None and _ENSG_RE.match(s):
        base = s.split(".", 1)[0]
        return gtf_map["ensg_to_symbol"].get(base, s)
    s = _PREFIX_RE.sub("", s).lstrip("_")
    s = _VERSION_RE.sub("", s)
    s = _SUFFIX_RE.sub(r"-\1\2", s)
    return s


def clean_adata_var_names(adata, rescue_protein_coding: bool = True,
                           resolve_ensg: bool = True):
    """Apply full cleaning pipeline to `adata.var_names`.

    Returns
    -------
    (cleaned_adata, n_noise_dropped)
        New AnnData view with normalised names + noise + duplicates removed.

    Notes
    -----
    `resolve_ensg=True` (default) lazily loads the GTF map only when the
    first 32 var names look like Ensembl IDs — so the 89% of HEST samples
    that ship HGNC don't pay the GTF-load cost.  Critical for ~10% of HEST
    (Heart 60 + Breast 68) whose var names are raw ENSG; without this
    they'd have ≈ 0 % vocab match (silent data loss).
    """
    gtf_map = _maybe_load_gtf(adata, resolve_ensg)

    # 1) symbol normalise — twice so versions/prefix interact cleanly.
    cleaned = [clean_symbol(v, gtf_map) for v in adata.var_names]
    adata = adata[:, [bool(c) for c in cleaned]].copy()
    cleaned = [clean_symbol(v, gtf_map) for v in adata.var_names]
    adata.var_names = np.array(cleaned, dtype=object)

    # 2) dedup
    adata = adata[:, ~adata.var_names.duplicated()].copy()

    # 3) drop unmapped ENSG (when GTF was loaded but couldn't resolve them)
    if gtf_map is not None:
        still_ensg = np.array(
            [str(v).upper().startswith(("ENSG", "ENSMUSG")) for v in adata.var_names],
            dtype=bool,
        )
        if still_ensg.any():
            log.info(f"clean_adata_var_names: dropped {int(still_ensg.sum())} unmapped "
                      f"ENSG (not in current GTF — likely deprecated)")
            adata = adata[:, ~still_ensg].copy()

    # 4) noise-pattern drop, with optional protein_coding rescue.
    n_before = adata.n_vars
    adata = _drop_noise(adata, rescue_protein_coding=rescue_protein_coding)
    return adata, n_before - adata.n_vars


# ── Internals ───────────────────────────────────────────────────────────────

def _maybe_load_gtf(adata, resolve_ensg: bool):
    """Lazily load the GTF map IFF the input has Ensembl-looking names."""
    if not resolve_ensg:
        return None
    sample_vn = list(map(str, adata.var_names[:32]))
    if not any(_ENSG_RE.match(v.upper().strip()) for v in sample_vn):
        return None
    try:
        from .gene_symbols import load_gtf_symbol_map
        return load_gtf_symbol_map()
    except Exception as e:
        log.warning(f"GTF load for ENSG resolution failed: {e}")
        return None


def _drop_noise(adata, rescue_protein_coding: bool):
    """Drop var names matching `_NOISE_RE`, optionally rescuing PC genes."""
    names = adata.var_names.to_series()
    hit = names.str.match(_NOISE_RE).values            # True = pattern hit

    if rescue_protein_coding and hit.any():
        try:
            from .gene_symbols import load_gtf_symbol_map
            type_map = load_gtf_symbol_map()["symbol_to_gene_type"]
            rescued = np.zeros_like(hit)
            for i, (n, h) in enumerate(zip(names.values, hit)):
                if not h:
                    continue
                s = str(n).upper()
                if type_map.get(s, "") in _ALLOW_TYPES and not _NEVER_RESCUE_RE.match(s):
                    rescued[i] = True
            n_rescued = int(rescued.sum())
            if n_rescued:
                log.info(f"clean_adata_var_names: rescued {n_rescued} pattern-hit "
                          f"protein-coding genes (e.g. TP53/MMP/HSP families)")
                hit = hit & (~rescued)
        except Exception as e:
            log.warning(f"GTF rescue skipped: {e}")

    return adata[:, ~hit].copy()
