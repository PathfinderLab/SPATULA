"""Align-JEPA + spatial-JEPA — with optional in-batch CLIP-negative term.

Branches:
  align-JEPA   : student z_image → pred_i2t → ≈ teacher z_tx  (and reverse)
  spatial-JEPA : student aggregated K-NN shared latent → pred_spatial → teacher shared latent
  (optional) negatives : symmetric InfoNCE on (z_image, z_tx) — controlled by
                          experiment.align.negative_weight. Adds the missing
                          "competing views" signal that pure cosine-to-EMA lacks.

Design fixes vs. previous version:
  * Teacher modules are switched to **eval() mode** inside `_teacher_encode`
    so BatchNorm uses running stats (was a known BYOL bug — train-mode BN
    in the target collapses the representation).
  * VICReg var/cov regularization on student z's stays in (var/cov weights
    higher than before since JEPA has no in-batch negatives by default).
  * Anti-collapse diagnostic logged each step: diagonal vs off-diagonal
    cosine similarity — a clear collapse signature when diag ≈ off-diag.
"""
from __future__ import annotations
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import AlignLoss


@torch.no_grad()
def _ema_update(student_mod: nn.Module, teacher_mod: nn.Module, m: float) -> None:
    for ps, pt in zip(student_mod.parameters(), teacher_mod.parameters()):
        pt.data.mul_(m).add_(ps.data, alpha=1.0 - m)


def _var_cov_reg(z: torch.Tensor, var_target: float = 1.0):
    zc = z - z.mean(0, keepdim=True)
    std = torch.sqrt(zc.var(0) + 1e-4)
    var_loss = F.relu(var_target - std).mean()
    B, D = zc.shape
    cov = (zc.T @ zc) / max(B - 1, 1)
    off = cov - torch.diag(torch.diagonal(cov))
    cov_loss = off.pow(2).sum() / D
    return var_loss, cov_loss


class _Predictor(nn.Module):
    def __init__(self, in_dim, hidden, out, n_layers=2):
        super().__init__()
        layers = []
        cur = in_dim
        for _ in range(n_layers - 1):
            layers += [nn.Linear(cur, hidden), nn.GELU()]
            cur = hidden
        layers += [nn.Linear(cur, out)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class JEPAAlign(AlignLoss):
    def __init__(self, cfg, model):
        super().__init__(cfg, model)
        ocfg = cfg["experiment"]["align"]
        D = model.proj_dim
        h = ocfg.get("pred_hidden", 512)
        nl = ocfg.get("pred_layers", 2)

        # Light teacher copies: only adapter + tx_encoder + projectors.
        s_img_enc = model.image_encoder
        self.t_img_norm = copy.deepcopy(s_img_enc.norm)
        self.t_img_adapter = copy.deepcopy(s_img_enc.adapter)
        self.t_tx_encoder = copy.deepcopy(model.tx_encoder)
        self.t_image_proj = copy.deepcopy(model.image_proj)
        self.t_tx_proj = copy.deepcopy(model.tx_proj)
        for m in (self.t_img_norm, self.t_img_adapter, self.t_tx_encoder,
                  self.t_image_proj, self.t_tx_proj):
            for p in m.parameters():
                p.requires_grad_(False)

        self.m = ocfg.get("ema_momentum", 0.9995)

        self.use_align = ocfg.get("use_align_jepa", True)
        self.use_spatial = ocfg.get("use_spatial_jepa", True)
        if self.use_align:
            self.pred_i2t = _Predictor(D, h, D, nl)
            self.pred_t2i = _Predictor(D, h, D, nl)
        if self.use_spatial:
            self.pred_spatial = _Predictor(D, h, D, nl)

        self.align_weight = ocfg.get("align_branch_weight", 1.0)
        self.spatial_weight = ocfg.get("spatial_weight", 0.5)
        self.var_weight = ocfg.get("var_weight", 4.0)
        self.cov_weight = ocfg.get("cov_weight", 0.5)
        self.var_target = ocfg.get("var_target", 1.0)

        # Optional in-batch CLIP-style negatives. Adds the contrastive signal
        # that pure JEPA lacks.  0 → vanilla JEPA, 0.2 → small contrastive
        # regularizer, 1.0 → CLIP-JEPA hybrid.
        self.negative_weight = ocfg.get("negative_weight", 0.2)
        init_T = ocfg.get("nce_temperature_init", 0.07)
        if self.negative_weight > 0:
            self.nce_log_scale = nn.Parameter(torch.tensor(math.log(1.0 / init_T)))

    def _student(self):
        return self.__dict__["model"]

    @torch.no_grad()
    def _teacher_encode(self, batch):
        student = self._student()
        # ── BN / Dropout → eval mode on teacher modules.  Critical for stable
        # targets: BN in train mode would update running stats every step,
        # destroying the EMA-slow-target property (BYOL/DINO).
        prev_states = []
        for m in (self.t_img_norm, self.t_img_adapter, self.t_tx_encoder,
                  self.t_image_proj, self.t_tx_proj):
            prev_states.append(m.training)
            m.eval()
        try:
            s_img_enc = student.image_encoder
            if getattr(s_img_enc, "has_trunk", False):
                x = batch["image_raw"]
                if s_img_enc.preproc is not None:
                    x = s_img_enc.preproc(x)
                trunk_feat = s_img_enc.trunk(x)
                if s_img_enc.trunk_adapter is not None:
                    trunk_feat = s_img_enc.trunk_adapter(trunk_feat)
            else:
                trunk_feat = batch["image"]
            trunk_feat = trunk_feat.detach()
            t_h_img = self.t_img_adapter(self.t_img_norm(trunk_feat))
            t_z_img = self.t_image_proj(t_h_img)
            # tx_encoder now returns a dict {h_tx, per_token, mask?} — extract h_tx.
            t_tx_out = self.t_tx_encoder(batch.get("tx_latent"), batch.get("hvg"))
            t_h_tx = t_tx_out["h_tx"] if isinstance(t_tx_out, dict) else t_tx_out
            t_z_tx = self.t_tx_proj(t_h_tx)
            return {"z_image": t_z_img, "z_tx": t_z_tx,
                    "z_shared": 0.5 * (t_z_img + t_z_tx)}
        finally:
            for m, s in zip((self.t_img_norm, self.t_img_adapter, self.t_tx_encoder,
                              self.t_image_proj, self.t_tx_proj), prev_states):
                m.train(s)

    @staticmethod
    def _cos_dist(p_raw, t_norm):
        p = F.normalize(p_raw, dim=-1)
        return (2 - 2 * (p * t_norm).sum(-1)).mean()

    @staticmethod
    def _neighbor_aggregate(s_lat, batch):
        nb = batch["neighbors"]
        sample_idx = batch["sample_idx"].to(torch.int64)
        spot_idx = batch["spot_idx"].to(torch.int64)
        key = (sample_idx << 32) | spot_idx
        nb_key = (sample_idx.unsqueeze(1) << 32) | nb.to(torch.int64)
        match = nb_key.unsqueeze(-1) == key.unsqueeze(0).unsqueeze(0)
        found = match.any(dim=-1)
        pos = match.float().argmax(dim=-1)
        nb_lat = s_lat[pos] * found.unsqueeze(-1).float()
        cnt = found.sum(dim=-1).float()
        agg = nb_lat.sum(dim=1) / cnt.clamp(min=1.0).unsqueeze(-1)
        return agg, cnt > 0

    def forward(self, model_out, batch):
        s_img = model_out["z_image"]
        s_tx = model_out["z_tx"]
        s_shared = model_out["z_shared"]

        with torch.no_grad():
            t = self._teacher_encode(batch)
            t_img_n = F.normalize(t["z_image"], dim=-1)
            t_tx_n = F.normalize(t["z_tx"], dim=-1)
            t_shared_n = F.normalize(t["z_shared"], dim=-1)

        loss = torch.zeros((), device=s_img.device)
        log = {}

        if self.use_align:
            l_i2t = self._cos_dist(self.pred_i2t(s_img), t_tx_n)
            l_t2i = self._cos_dist(self.pred_t2i(s_tx), t_img_n)
            l_align = 0.5 * (l_i2t + l_t2i)
            loss = loss + self.align_weight * l_align
            log["align/i2t"] = l_i2t.detach()
            log["align/t2i"] = l_t2i.detach()

        if self.use_spatial and "neighbors" in batch:
            ctx, hit = self._neighbor_aggregate(s_shared, batch)
            p = F.normalize(self.pred_spatial(ctx), dim=-1)
            per_row = 2 - 2 * (p * t_shared_n).sum(-1)
            mask_f = hit.float()
            denom = mask_f.sum().clamp(min=1.0)
            l_sp = (per_row * mask_f).sum() / denom
            loss = loss + self.spatial_weight * l_sp
            log["align/spatial"] = l_sp.detach()
            log["align/spatial_hit_ratio"] = mask_f.mean().detach()

        if self.var_weight > 0 or self.cov_weight > 0:
            v_i, c_i = _var_cov_reg(s_img, self.var_target)
            v_t, c_t = _var_cov_reg(s_tx, self.var_target)
            v_loss = 0.5 * (v_i + v_t)
            c_loss = 0.5 * (c_i + c_t)
            loss = loss + self.var_weight * v_loss + self.cov_weight * c_loss
            log["align/var_reg"] = v_loss.detach()
            log["align/cov_reg"] = c_loss.detach()

        # ── Optional in-batch CLIP-style negatives ────────────────────
        # Pure cosine-to-EMA targets can collapse on small batches with no
        # negatives. A small InfoNCE term adds the "competing views" signal.
        if self.negative_weight > 0:
            zi = F.normalize(s_img, dim=-1)
            zt = F.normalize(s_tx, dim=-1)
            scale = self.nce_log_scale.exp().clamp(max=100.0)
            logits = scale * zi @ zt.t()
            tgt = torch.arange(zi.size(0), device=zi.device)
            l_i2t = F.cross_entropy(logits, tgt)
            l_t2i = F.cross_entropy(logits.t(), tgt)
            l_nce = 0.5 * (l_i2t + l_t2i)
            loss = loss + self.negative_weight * l_nce
            log["align/nce"] = l_nce.detach()
            log["align/nce_temp"] = (1.0 / scale).detach()

        # ── Collapse diagnostic ──────────────────────────────────────
        # On a healthy batch: diagonal cosine sim (paired) >> off-diagonal.
        # When they converge, the model is collapsing.
        with torch.no_grad():
            zi_n = F.normalize(s_img, dim=-1)
            zt_n = F.normalize(s_tx, dim=-1)
            sim = zi_n @ zt_n.t()
            B = sim.size(0)
            diag = sim.diag().mean()
            off = (sim.sum() - sim.diag().sum()) / max(1, B * (B - 1))
            log["align/diag_cos"] = diag
            log["align/offdiag_cos"] = off
            log["align/diag_minus_off"] = diag - off

        log["align/loss"] = loss.detach()
        return loss, log

    def on_after_step(self) -> None:
        student = self._student()
        s_img_enc = student.image_encoder
        pairs = [
            (s_img_enc.norm, self.t_img_norm),
            (s_img_enc.adapter, self.t_img_adapter),
            (student.tx_encoder, self.t_tx_encoder),
            (student.image_proj, self.t_image_proj),
            (student.tx_proj, self.t_tx_proj),
        ]
        for sm, tm in pairs:
            _ema_update(sm, tm, self.m)
