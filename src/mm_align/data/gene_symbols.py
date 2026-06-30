"""GTF-based gene symbol canonicalization + ENSG mapping.

Parses /workspace/assets/gencode.v49.annotation.gtf (or any GENCODE GTF) once
into:
  - ensg → symbol            (strip the .X version suffix; one record per gene)
  - symbol → ensg            (last-write-wins; ambiguous symbols → use the
                              GENCODE canonical chromosome — we keep the first
                              chr1..22/X/Y/M record per symbol)
  - symbol → gene_type       ("protein_coding", "lncRNA", "pseudogene", ...)
  - set of valid HGNC symbols (everything that appears under "gene_name")

Cached as JSON at <gtf_path>.symbol_map.json so repeated calls are O(load).

Public functions:
  load_gtf_symbol_map(gtf_path) -> dict {ensg_to_symbol, symbol_to_ensg,
                                         symbol_to_gene_type, valid_symbols}
  canonicalize_symbol(name, gmap) -> str   # ENSG → symbol if found; else name
  classify_symbol(name, gmap) -> str       # "valid_hgnc" | "ensg_unresolved" |
                                            "noise" | "unknown"
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

# ENSG with optional version suffix
_ENSG_RE = re.compile(r"^ENSG\d+(\.\d+)?$", re.IGNORECASE)
# Standard primary chromosomes — drop ALT contigs / scaffolds
_PRIMARY_CHR = {f"chr{i}" for i in list(range(1, 23))} | {"chrX", "chrY", "chrM", "chrMT"}


def _strip_version(ensg: str) -> str:
    """ENSG00000123456.7 → ENSG00000123456"""
    return ensg.split(".", 1)[0].upper()


def _parse_gtf(gtf_path: Path) -> dict:
    """Stream-parse a GENCODE GTF, return the 4 maps described in the module doc."""
    ensg_to_symbol: dict[str, str] = {}
    symbol_to_ensg: dict[str, str] = {}
    symbol_to_gene_type: dict[str, str] = {}
    valid_symbols: set[str] = set()

    # Lightweight regex for the attributes column.  GTF uses ; to separate.
    attr_re = re.compile(r'(\w+)\s+"([^"]+)"')
    n_lines = 0
    n_gene = 0
    with open(gtf_path, "rt") as f:
        for line in f:
            if not line or line[0] == "#":
                continue
            n_lines += 1
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 9 or cols[2] != "gene":
                continue
            chrom = cols[0]
            attrs = dict(attr_re.findall(cols[8]))
            ensg = attrs.get("gene_id", "")
            symbol = attrs.get("gene_name", "")
            gene_type = attrs.get("gene_type", "")
            if not (ensg and symbol):
                continue
            ensg = _strip_version(ensg)
            symbol_u = symbol.upper()
            ensg_to_symbol[ensg] = symbol_u
            valid_symbols.add(symbol_u)
            # Prefer primary-chromosome record for symbol → ensg / gene_type
            if symbol_u not in symbol_to_ensg or chrom in _PRIMARY_CHR:
                symbol_to_ensg[symbol_u] = ensg
                symbol_to_gene_type[symbol_u] = gene_type
            n_gene += 1
    return {
        "ensg_to_symbol": ensg_to_symbol,
        "symbol_to_ensg": symbol_to_ensg,
        "symbol_to_gene_type": symbol_to_gene_type,
        "valid_symbols": sorted(valid_symbols),
        "_meta": {"lines": n_lines, "genes": n_gene},
    }


@lru_cache(maxsize=4)
def load_gtf_symbol_map(gtf_path: str | Path = "/workspace/assets/gencode.v49.annotation.gtf") -> dict:
    """Load (or build + cache) the GTF symbol map.

    Side-effect: writes <gtf_path>.symbol_map.json next to the GTF so subsequent
    calls in any process don't re-parse the 7M-line GTF (parsing takes ~30s).
    """
    gtf_path = Path(gtf_path)
    cache = gtf_path.with_suffix(gtf_path.suffix + ".symbol_map.json")
    if cache.exists():
        with open(cache) as fh:
            payload = json.load(fh)
        # rehydrate valid_symbols as a set
        payload["valid_symbols"] = set(payload["valid_symbols"])
        return payload
    if not gtf_path.exists():
        raise FileNotFoundError(f"GTF not found: {gtf_path}")
    payload = _parse_gtf(gtf_path)
    # Persist for next process
    try:
        with open(cache, "w") as fh:
            json.dump(payload, fh)
    except OSError:
        pass  # caching is best-effort
    payload["valid_symbols"] = set(payload["valid_symbols"])
    return payload


def canonicalize_symbol(name: str, gmap: dict) -> str:
    """If `name` looks like an ENSG, resolve it via the GTF; else return name uppercased.
    Does NOT apply noise-pattern filtering."""
    if not isinstance(name, str) or not name:
        return ""
    s = name.strip().upper()
    if _ENSG_RE.match(s):
        return gmap["ensg_to_symbol"].get(_strip_version(s), s)
    return s


def classify_symbol(name: str, gmap: dict, noise_re: "re.Pattern | None" = None) -> str:
    """One-shot classifier:
      "valid_hgnc"        — recognised symbol in GTF
      "ensg_unresolved"   — ENSG[…] not found in GTF
      "noise"             — matches the noise pattern (caller supplies)
      "unknown"           — non-ENSG, non-noise, not in GTF
    """
    if not name:
        return "unknown"
    s = name.strip().upper()
    if _ENSG_RE.match(s):
        return "valid_hgnc" if _strip_version(s) in gmap["ensg_to_symbol"] else "ensg_unresolved"
    if noise_re is not None and noise_re.match(s):
        return "noise"
    return "valid_hgnc" if s in gmap["valid_symbols"] else "unknown"
