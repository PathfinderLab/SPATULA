"""Spatial Predictive JEPA — Stage 1.5 foundation objective.

Student-Teacher latent prediction at masked SPOT positions.  The masked
anchor cell's whole token (spot_tx + spot_img + region_tx + region_img,
all fused into one cell-level token by SpotFuser) is replaced with the
learnable `mask_embed` before the student runs; the teacher sees the full
sequence.  Loss is smooth-L1 (or cosine) between student and teacher at
masked positions only.

Two masking strategies:
    random — Bernoulli per cell (legacy)
    block  — ST-JEPA-style: grow connected blocks of `block_size` cells
             via random walks on the spatial graph and mask them together.
             This forces the model to predict each masked cell from spots
             that are spatially adjacent but OUTSIDE the masked block,
             which is much harder than scattered i.i.d. masking and is the
             ST-JEPA paper's recommended default.

Teacher = EMA of student; updated by `on_after_step()` after the optimizer step.
"""
from __future__ import annotations
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def _ema_update(student: nn.Module, teacher: nn.Module, m: float):
    for ps, pt in zip(student.parameters(), teacher.parameters()):
        pt.data.mul_(m).add_(ps.data, alpha=1 - m)
    for bs, bt in zip(student.buffers(), teacher.buffers()):
        bt.data.copy_(bs.data)


class SpatialJEPAObjective(nn.Module):
    """Spatial Predictive JEPA loss + EMA teacher update.

    `mask_target` semantics:
        `spot`    — only spot tokens are masked.  In separate-token mode this
                    is the I-JEPA-faithful "region as visible context predicts
                    masked spot latent" — the encoder's main candidate.
        `region`  — only region tokens are masked.
        `both`    — same anchor's spot AND region get masked together.

    In fused-token mode there is only one stream, so `spot` and `both` mean
    the same thing (the cell-level token gets masked); `region` is rejected.
    """

    def __init__(self, student_encoder: nn.Module, *,
                 mask_ratio: float = 0.30,
                 mask_strategy: str = "block",
                 mask_target: str = "spot",
                 block_size: int = 8,
                 loss_kind: str = "smooth_l1",
                 smoothness_weight: float = 0.0,
                 ema_momentum: float = 0.999):
        super().__init__()
        assert loss_kind in ("smooth_l1", "cosine")
        assert mask_strategy in ("random", "block")
        assert mask_target in ("spot", "region", "both")
        token_mode = getattr(student_encoder, "token_mode", "fused")
        if token_mode == "fused" and mask_target == "region":
            raise ValueError(
                "mask_target='region' is meaningful only when "
                "token_mode='separate' (fused mode has no region-only stream).")
        self.mask_ratio = float(mask_ratio)
        self.mask_strategy = mask_strategy
        self.mask_target = mask_target
        self.block_size = int(block_size)
        self.loss_kind = loss_kind
        self.smoothness_weight = float(smoothness_weight)
        self.m = float(ema_momentum)
        self.student = student_encoder
        self.teacher = copy.deepcopy(student_encoder)
        for p in self.teacher.parameters():
            p.requires_grad_(False)

    # ── mask sampling ──────────────────────────────────────────────────────

    def _sample_mask(self, n_spots: int, device, *,
                       edge_index: torch.Tensor | None = None,
                       subgraph_id: torch.Tensor | None = None) -> torch.Tensor:
        if self.mask_ratio <= 0:
            return torch.zeros(n_spots, dtype=torch.bool, device=device)
        if self.mask_strategy == "random" or edge_index is None:
            return torch.rand(n_spots, device=device) < self.mask_ratio
        return self._block_mask(n_spots, device, edge_index, subgraph_id)

    def _block_mask(self, n_spots: int, device,
                     edge_index: torch.Tensor,
                     subgraph_id: torch.Tensor | None) -> torch.Tensor:
        """ST-JEPA-style: connected blocks of cells, sampled per-subgraph."""
        # Build adjacency dict once (CPU; subgraph_size is small per step).
        ei = edge_index.detach().cpu().numpy()
        adj: dict[int, list[int]] = {}
        for s, d in zip(ei[0], ei[1]):
            adj.setdefault(int(s), []).append(int(d))
        # Group nodes by subgraph so we don't cross subgraph boundaries.
        if subgraph_id is None:
            groups = {0: list(range(n_spots))}
        else:
            groups: dict[int, list[int]] = {}
            for i, sg in enumerate(subgraph_id.cpu().tolist()):
                groups.setdefault(int(sg), []).append(i)

        mask = torch.zeros(n_spots, dtype=torch.bool, device=device)
        for nodes in groups.values():
            target = max(1, int(round(len(nodes) * self.mask_ratio)))
            picked: set[int] = set()
            # Hard cap on outer loops so a small subgraph with no edges can't
            # spin forever; falls back to whatever we picked so far.
            attempts = 0
            while len(picked) < target and attempts < target * 8:
                attempts += 1
                seed = int(nodes[torch.randint(len(nodes), (1,), device="cpu").item()])
                # BFS up to block_size cells.
                frontier = [seed]
                grown = 0
                while frontier and grown < self.block_size and len(picked) < target:
                    cur = frontier.pop()
                    if cur in picked:
                        continue
                    picked.add(cur); grown += 1
                    nbs = adj.get(cur, [])
                    if nbs:
                        # Randomise order so the walk isn't deterministic.
                        perm = torch.randperm(len(nbs)).tolist()
                        frontier.extend(nbs[i] for i in perm)
            for p in picked:
                mask[p] = True
        return mask

    # ── losses ─────────────────────────────────────────────────────────────

    def _smooth_l1(self, pred, tgt):
        return F.smooth_l1_loss(pred, tgt, reduction="mean", beta=1.0)

    def _cosine(self, pred, tgt):
        return (2 - 2 * (F.normalize(pred, dim=-1) * F.normalize(tgt, dim=-1)).sum(-1)).mean()

    # ── forward ────────────────────────────────────────────────────────────

    def forward(self, batch: dict) -> tuple[torch.Tensor, dict]:
        """batch dict must contain:
            h_tx          (N, tx_dim)        — Stage-1 frozen tx CLS, anchor spots
            h_img         (N, img_dim)|None  — UNI features at anchor spots
            xy            (N, 2)             — normalised coords
            edge_index    (2, E)             — KNN/radius/grid edges
        Optional region tokens (when SpatialEncoder.fuse_region=True):
            h_region_tx   (N, tx_dim)
            h_region_img  (N, img_dim)|None
        Optional batching aids:
            subgraph_id   (N,) long          — for per-subgraph block masking
        """
        h_tx = batch["h_tx"]; h_img = batch.get("h_img"); xy = batch["xy"]
        edge_index = batch["edge_index"]
        h_region_tx = batch.get("h_region_tx")
        h_region_img = batch.get("h_region_img")
        subgraph_id = batch.get("subgraph_id")

        N = h_tx.shape[0]
        token_mode = getattr(self.student, "token_mode", "fused")

        # Sample anchor-level mask (size N).  In separate mode we then expand
        # to a 2N stream mask according to mask_target; in fused mode the
        # encoder consumes (N,) directly.
        anchor_mask = self._sample_mask(N, h_tx.device,
                                          edge_index=edge_index,
                                          subgraph_id=subgraph_id)
        if token_mode == "separate":
            zero = torch.zeros_like(anchor_mask)
            if self.mask_target == "spot":
                mask = torch.cat([anchor_mask, zero], dim=0)
            elif self.mask_target == "region":
                mask = torch.cat([zero, anchor_mask], dim=0)
            else:                                                 # both
                mask = torch.cat([anchor_mask, anchor_mask], dim=0)
        else:
            mask = anchor_mask

        z_student = self.student(h_tx, h_img, xy, edge_index, mask=mask,
                                   h_region_tx=h_region_tx,
                                   h_region_img=h_region_img)
        with torch.no_grad():
            z_teacher = self.teacher(h_tx, h_img, xy, edge_index, mask=None,
                                      h_region_tx=h_region_tx,
                                      h_region_img=h_region_img)

        if mask.any():
            pred = z_student[mask]
            tgt = z_teacher[mask].detach()
            l_jepa = self._smooth_l1(pred, tgt) if self.loss_kind == "smooth_l1" \
                else self._cosine(pred, tgt)
        else:
            l_jepa = z_student.sum() * 0.0       # no-op for empty mask

        log = {"spatial/jepa_loss": l_jepa.detach(),
               "spatial/mask_frac": float(mask.float().mean()),
               "spatial/mask_strategy": float(1.0 if self.mask_strategy == "block" else 0.0),
               "spatial/token_mode": float(1.0 if token_mode == "separate" else 0.0)}

        loss = l_jepa
        if self.smoothness_weight > 0:
            # Smoothness aux applied on the spot stream only (positions 0..N-1).
            z_spot = z_student[:N]
            src, dst = edge_index[0], edge_index[1]
            edge_l2 = (z_spot[src] - z_spot[dst]).pow(2).sum(-1).mean()
            loss = loss + self.smoothness_weight * edge_l2
            log["spatial/smooth_l2"] = edge_l2.detach()

        log["spatial/loss"] = loss.detach()
        return loss, log

    def on_after_step(self) -> None:
        _ema_update(self.student, self.teacher, self.m)
