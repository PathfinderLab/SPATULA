"""mm_align.models — public API.

Subdomains:
    .tx        — gene/transcriptomics encoders
    .image     — image encoders + foundation models
    .spatial   — spatial-context encoder
    .alignment — MMAligner, projection heads, decoders
"""
from .image.encoder import (
    FeatureImageEncoder,
    UNIImageEncoder,
    build_image_encoder,
)
from .image.foundation import build_foundation
from .image.uni import build_uni, apply_freeze, UNINormalize
from .tx.factory import build_tx_encoder, TranscriptEncoder
from .tx.top_hvg_gene import TopHVGGeneEncoder
from .tx.hvg_tokenizer import HVGTokenEncoder
from .spatial.encoder import SpatialEncoder
from .alignment.aligner import MMAligner
from .alignment.heads import MLPProjector, MaskedGeneHead
from .alignment.decoders import ImgEmbDecoder, GeneEmbDecoder

# Back-compat alias (older code imported ImageEncoder = FeatureImageEncoder).
ImageEncoder = FeatureImageEncoder

__all__ = [
    "FeatureImageEncoder", "UNIImageEncoder", "build_image_encoder",
    "TranscriptEncoder", "TopHVGGeneEncoder", "HVGTokenEncoder", "build_tx_encoder",
    "SpatialEncoder",
    "build_foundation", "build_uni", "apply_freeze", "UNINormalize",
    "MLPProjector", "MaskedGeneHead",
    "ImgEmbDecoder", "GeneEmbDecoder",
    "MMAligner",
    "ImageEncoder",
]
