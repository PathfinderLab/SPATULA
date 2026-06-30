from .base import AlignLoss
from .clip import CLIPAlign
from .barlow import BarlowAlign
from .cca import CCAAlign
from .s2l import S2LAlign
from .jepa import JEPAAlign


def build_align_loss(method: str, cfg: dict, model) -> AlignLoss:
    method = method.lower()
    if method == "clip":
        return CLIPAlign(cfg, model)
    if method in ("bt", "barlow", "barlow_twins"):
        return BarlowAlign(cfg, model)
    if method == "cca":
        return CCAAlign(cfg, model)
    if method in ("s2l", "soft_clip"):
        return S2LAlign(cfg, model)
    if method == "jepa":
        return JEPAAlign(cfg, model)
    raise ValueError(f"Unknown align method: {method}")


__all__ = ["AlignLoss", "build_align_loss",
           "CLIPAlign", "BarlowAlign", "CCAAlign", "S2LAlign", "JEPAAlign"]
