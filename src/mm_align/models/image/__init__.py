"""Image encoders — UNI ViT backbone + feature MLP + foundation model registry."""
from .encoder import FeatureImageEncoder, UNIImageEncoder, build_image_encoder
from .foundation import build_foundation
from .uni import build_uni, apply_freeze, UNINormalize

__all__ = ["FeatureImageEncoder", "UNIImageEncoder", "build_image_encoder",
           "build_foundation", "build_uni", "apply_freeze", "UNINormalize"]
