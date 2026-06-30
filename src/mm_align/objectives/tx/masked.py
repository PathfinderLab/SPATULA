"""Tx-side self-supervised losses for the learnable gene encoders.

Two encoder kinds trigger this module:

  - `hvg_tokenizer`  (legacy lightweight version) → masked-recon + masked-JEPA
  - `top_hvg_gene`   (SEAL/spatula style)        → masked-symbol CE
                                                  + masked-value MSE
                                                  + optional masked-JEPA on
                                                    per-token EMA-teacher latents
                                                  + optional DINO-style
                                                    spot-level consistency.

In both cases the EMA teacher is a deep copy of the tx encoder.  The teacher's
gene/value/jepa heads are not used; we only need clean contextualised token/spot
embeddings as stop-gradient targets.
"""
from __future__ import annotations
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def _ema_update(student_mod: nn.Module, teacher_mod: nn.Module, m: float) -> None:
    for ps, pt in zip(student_mod.parameters(), teacher_mod.parameters()):
        pt.data.mul_(m).add_(ps.data, alpha=1.0 - m)
    for bs, bt in zip(student_mod.buffers(), teacher_mod.buffers()):
        bt.data.copy_(bs.data)


# NOTE: `_sinkhorn_knopp` was removed (2026-06-17) — sinkhorn DINO mode on
# raw 512-d h_tx is collapse-prone without a dedicated projection head.
# See README / docs/design/stage1_objective.md for re-enable plan.




def _off_diagonal(x: torch.Tensor) -> torch.Tensor:
    n, m = x.shape
    if n != m:
        return x.new_empty((0,))
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def _vicreg_var_cov_loss(
    z: torch.Tensor,
    *,
    gamma: float = 1.0,
    eps: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """VICReg variance/covariance anti-collapse terms for one embedding batch.

    The JEPA prediction loss already supplies the invariance/alignment term.
    These terms keep the learned representation non-collapsed and reduce
    redundant dimensions, matching the transcriptomics JEPA references more
    closely than nearest-neighbor spacing alone.
    """
    if z.ndim != 2 or z.shape[0] < 2:
        zero = z.new_zeros(())
        return zero, zero
    z = z.float()
    std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)
    var_loss = F.relu(float(gamma) - std).mean()
    z = z - z.mean(dim=0, keepdim=True)
    cov = (z.T @ z) / max(1, z.shape[0] - 1)
    cov_loss = _off_diagonal(cov).pow(2).sum() / z.shape[1]
    return var_loss, cov_loss

def _koleo_loss(z: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Kozachenko-Leonenko entropy regularizer on batch embeddings.

    Maximises local spacing by penalising small nearest-neighbour distances.
    """
    if z.ndim != 2 or z.shape[0] < 2:
        return z.new_zeros(())
    z = F.normalize(z.float(), dim=-1)
    dist = torch.cdist(z, z, p=2)
    eye = torch.eye(dist.shape[0], device=dist.device, dtype=torch.bool)
    dist = dist.masked_fill(eye, float("inf"))
    nn_dist = dist.min(dim=1).values.clamp_min(eps)
    return -torch.log(nn_dist).mean()


class MaskedTxLosses(nn.Module):
    def __init__(self, *, model, ema_momentum: float = 0.999,
                 enable_jepa: bool = True,
                 enable_dino_consistency: bool = False,
                 symbol_weight: float = 1.0,
                 value_weight: float = 0.5,
                 jepa_weight: float = 1.0,
                 enable_view_jepa: bool = False,
                 view_jepa_weight: float = 0.0,
                 view_jepa_loss: str = "smooth_l1",
                 view_jepa_hidden_dim: int = 1024,
                 view_jepa_warmup_epochs: int = 0,
                 view_jepa_ramp_epochs: int = 0,
                 enable_multi_chunk_jepa: bool = False,
                 multi_chunk_weight: float = 0.0,
                 multi_chunk_n_chunks: int = 4,
                 multi_chunk_len: int = 256,
                 multi_chunk_loss: str = "smooth_l1",
                 multi_chunk_target: str = "target_chunk",
                 multi_chunk_dynamic: bool = True,
                 multi_chunk_target_chunks: int | str = 2,
                 multi_chunk_target_scale: tuple[float, float] = (0.15, 0.25),
                 multi_chunk_context_scale: tuple[float, float] = (0.45, 0.65),
                 multi_chunk_hidden_dim: int = 1024,
                 multi_chunk_target_id_scale: float = 0.5,
                 multi_chunk_koleo_weight: float = 0.0,
                 multi_chunk_regularizer: str = "none",
                 multi_chunk_vicreg_var_weight: float = 1.0,
                 multi_chunk_vicreg_cov_weight: float = 1.0,
                 multi_chunk_vicreg_gamma: float = 1.0,
                 multi_chunk_warmup_epochs: int = 0,
                 multi_chunk_ramp_epochs: int = 0,
                 dino_weight: float = 0.0,
                 dino_loss: str = "cosine",
                 dino_student_temp: float = 0.1,
                 dino_teacher_temp: float = 0.04,
                 sinkhorn_iterations: int = 3,    # unused (sinkhorn disabled); kept for config back-compat
                 dino_warmup_epochs: int = 0,
                 dino_ramp_epochs: int = 0,
                 koleo_weight: float = 0.0,
                 koleo_eps: float = 1e-6,
                 koleo_warmup_epochs: int = 0,
                 koleo_ramp_epochs: int = 0,
                 masking_obj: str = "both"):
        """
        Parameters
        ----------
        masking_obj : "symbol" | "value" | "both"
            Which sub-loss to compute.  Aligned with SPATULA paper recommendation
            (primary = symbol modeling = MSM; value reconstruction is auxiliary).
            "symbol" → masked-symbol CE only (PDF default)
            "value"  → masked-value MSE only
            "both"   → both (legacy)
            Sub-weights (symbol_weight, value_weight) still gate further; setting
            either to 0 also disables that path.
        """
        super().__init__()
        object.__setattr__(self, "model", model)
        self.enable_jepa = enable_jepa
        self.enable_dino = enable_dino_consistency
        self.m = ema_momentum
        self.masking_obj = masking_obj
        # "obj" gate × weight gate — both must be on for a term to fire.
        self.use_symbol = masking_obj in ("symbol", "both") and symbol_weight > 0
        self.use_value = masking_obj in ("value", "both") and value_weight > 0
        self.w_symbol = symbol_weight
        self.w_value = value_weight
        self.w_jepa = jepa_weight
        self.enable_view_jepa = bool(enable_view_jepa)
        self.w_view_jepa = float(view_jepa_weight)
        if view_jepa_loss not in ("smooth_l1", "cosine"):
            raise ValueError(f"Unknown view_jepa_loss={view_jepa_loss!r}; use 'smooth_l1' or 'cosine'.")
        self.view_jepa_loss = view_jepa_loss
        self.view_jepa_warmup_epochs = max(0, int(view_jepa_warmup_epochs))
        self.view_jepa_ramp_epochs = max(0, int(view_jepa_ramp_epochs))
        self.enable_multi_chunk = bool(enable_multi_chunk_jepa)
        self.w_multi_chunk = float(multi_chunk_weight)
        self.multi_chunk_n = max(2, int(multi_chunk_n_chunks))
        self.multi_chunk_len = max(1, int(multi_chunk_len))
        self.multi_chunk_dynamic = bool(multi_chunk_dynamic)
        raw_target_chunks = str(multi_chunk_target_chunks).strip().lower()
        self.multi_chunk_target_chunks_auto = raw_target_chunks in ("auto", "dynamic", "ijepa")
        if self.multi_chunk_target_chunks_auto:
            self.multi_chunk_target_chunks = max(1, self.multi_chunk_n - 1)
        else:
            self.multi_chunk_target_chunks = max(1, int(multi_chunk_target_chunks))
        def _scale_pair(x, default):
            if isinstance(x, str):
                vals = [float(v.strip()) for v in x.split(",") if v.strip()]
            else:
                vals = list(x) if isinstance(x, (list, tuple)) else [float(x)]
            if len(vals) == 0:
                vals = list(default)
            if len(vals) == 1:
                vals = [vals[0], vals[0]]
            lo, hi = sorted((max(0.0, float(vals[0])), max(0.0, float(vals[1]))))
            return lo, max(lo, hi)
        self.multi_chunk_target_scale = _scale_pair(multi_chunk_target_scale, (0.15, 0.25))
        self.multi_chunk_context_scale = _scale_pair(multi_chunk_context_scale, (0.45, 0.65))
        self.multi_chunk_target_id_scale = float(multi_chunk_target_id_scale)
        self.w_multi_chunk_koleo = float(multi_chunk_koleo_weight)
        reg = str(multi_chunk_regularizer).strip().lower()
        if reg in ("", "off", "false", "0"):
            reg = "none"
        if reg not in ("none", "koleo", "vicreg"):
            raise ValueError(f"Unknown multi_chunk_regularizer={multi_chunk_regularizer!r}; use none|koleo|vicreg.")
        # Back-compat: old configs only had multi_chunk_koleo_weight.
        if reg == "none" and self.w_multi_chunk_koleo > 0:
            reg = "koleo"
        self.multi_chunk_regularizer = reg
        self.w_multi_chunk_vicreg_var = float(multi_chunk_vicreg_var_weight)
        self.w_multi_chunk_vicreg_cov = float(multi_chunk_vicreg_cov_weight)
        self.multi_chunk_vicreg_gamma = float(multi_chunk_vicreg_gamma)
        if multi_chunk_loss not in ("smooth_l1", "cosine"):
            raise ValueError(f"Unknown multi_chunk_loss={multi_chunk_loss!r}; use 'smooth_l1' or 'cosine'.")
        self.multi_chunk_loss = multi_chunk_loss
        if multi_chunk_target == "heldout_chunk":
            multi_chunk_target = "target_chunk"
        if multi_chunk_target not in ("target_chunk", "spot_aggregate"):
            raise ValueError(
                f"Unknown multi_chunk_target={multi_chunk_target!r}; "
                "use 'target_chunk' or 'spot_aggregate'."
            )
        self.multi_chunk_target = multi_chunk_target
        self.multi_chunk_warmup_epochs = max(0, int(multi_chunk_warmup_epochs))
        self.multi_chunk_ramp_epochs = max(0, int(multi_chunk_ramp_epochs))
        self.w_dino = dino_weight
        # NOTE: `sinkhorn` is currently DISABLED.  Without a dedicated
        # high-dim projection head, sinkhorn on the spot-level h_tx (e.g.
        # 512-d) collapses to entropy ≈ 2.3 ≪ log(512), making the teacher
        # probability mass concentrate on a few columns.  Re-enable only
        # after adding a 4096+ dim DINO projection head.
        if dino_loss in ("sinkhorn",):
            raise ValueError(
                f"dino_loss={dino_loss!r} is currently disabled — sinkhorn "
                "on the raw 512-d h_tx is collapse-prone (entropy 2.3 vs log(D)=6.2). "
                "Use 'cosine' (with centering, default) or 'smooth_l1'."
            )
        if dino_loss not in ("cosine", "smooth_l1"):
            raise ValueError(
                f"Unknown dino_loss={dino_loss!r}; use 'cosine' or 'smooth_l1'."
            )
        self.dino_loss = dino_loss
        self.dino_student_temp = float(dino_student_temp)
        self.dino_teacher_temp = float(dino_teacher_temp)
        self.sinkhorn_iterations = int(sinkhorn_iterations)
        self.dino_warmup_epochs = max(0, int(dino_warmup_epochs))
        self.dino_ramp_epochs = max(0, int(dino_ramp_epochs))
        self.w_koleo = float(koleo_weight)
        self.koleo_eps = float(koleo_eps)
        self.koleo_warmup_epochs = max(0, int(koleo_warmup_epochs))
        self.koleo_ramp_epochs = max(0, int(koleo_ramp_epochs))
        self.current_epoch = 0

        # The exact kind drives which heads we read.
        tx = model.tx_encoder
        self.tx_kind = getattr(tx, "kind", "unknown")
        if self.tx_kind not in ("hvg_tokenizer", "top_hvg_gene"):
            raise ValueError(
                f"MaskedTxLosses incompatible with tx_encoder.kind={self.tx_kind}; "
                "use 'hvg_tokenizer' or 'top_hvg_gene'."
            )

        h_dim = int(getattr(model, "embed_dim", getattr(model.tx_encoder, "out_dim", 512)))
        hidden = int(view_jepa_hidden_dim)
        self.view_jepa_predictor = None
        if self.enable_view_jepa:
            self.view_jepa_predictor = nn.Sequential(
                nn.LayerNorm(h_dim),
                nn.Linear(h_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, h_dim),
            )
        self.multi_chunk_predictor = None
        self.multi_chunk_target_query = None
        self.multi_chunk_target_id_proj = None
        if self.enable_multi_chunk:
            mc_hidden = int(multi_chunk_hidden_dim)
            token_dim = int(getattr(model.tx_encoder, "dim", h_dim))
            self.multi_chunk_predictor = nn.Sequential(
                nn.LayerNorm(h_dim),
                nn.Linear(h_dim, mc_hidden),
                nn.GELU(),
                nn.Linear(mc_hidden, h_dim),
            )
            # I-JEPA predicts multiple target blocks separately. The learned
            # ordinal query distinguishes target slots, while the gene-id query
            # tells the predictor *which* held-out gene block to predict, similar
            # to I-JEPA mask tokens carrying target position embeddings.
            self.multi_chunk_target_query = nn.Parameter(torch.zeros(self.multi_chunk_n, h_dim))
            nn.init.trunc_normal_(self.multi_chunk_target_query, std=0.02)
            self.multi_chunk_target_id_proj = nn.Sequential(
                nn.LayerNorm(token_dim),
                nn.Linear(token_dim, h_dim),
            )

        self.teacher = None
        if self.enable_jepa or self.enable_dino or self.enable_view_jepa or self.enable_multi_chunk:
            self.teacher = copy.deepcopy(model.tx_encoder)
            for p in self.teacher.parameters():
                p.requires_grad_(False)

    def _student(self):
        return self.__dict__["model"]

    # ------------------------------------------------------------------

    def set_epoch(self, epoch: int) -> None:
        """Expose the trainer epoch for auxiliary-loss warmup schedules."""
        self.current_epoch = int(epoch)

    def _warmup_scale(self, warmup_epochs: int, ramp_epochs: int) -> float:
        """Return 0 during warmup, then linearly ramp to 1.

        Epochs are 1-indexed in the trainer. For warmup=5, ramp=5:
        epochs 1-5 -> 0.0, epoch 6 -> 0.2, ..., epoch 10+ -> 1.0.
        """
        epoch = int(getattr(self, "current_epoch", 0))
        if epoch <= 0:
            return 1.0
        if epoch <= warmup_epochs:
            return 0.0
        if ramp_epochs <= 0:
            return 1.0
        return min(1.0, max(0.0, (epoch - warmup_epochs) / float(ramp_epochs)))

    def _student_forward_clean(self, hvg: torch.Tensor) -> dict:
        """Run student tx_encoder on a clean chunk view, without MSM masking.

        Gradients still flow through the encoder; eval mode is used only to
        disable internal mask sampling and dropout for stable chunk views.
        """
        tx = self._student().tx_encoder
        prev_training = tx.training
        prev_force_mask = getattr(tx, "_force_mask_in_eval", False)
        tx.eval()
        if hasattr(tx, "_force_mask_in_eval"):
            tx._force_mask_in_eval = False
        try:
            out = tx(None, hvg, mask=None)
        finally:
            tx.train(prev_training)
            if hasattr(tx, "_force_mask_in_eval"):
                tx._force_mask_in_eval = prev_force_mask
        return out

    def _sample_chunk_views(self, hvg: torch.Tensor, n_chunks: int, chunk_len: int) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
        """Create inference/eval chunk views from each spot's non-zero sequence."""
        chunks = [torch.zeros_like(hvg) for _ in range(n_chunks)]
        eff_lens = torch.zeros(hvg.shape[0], device=hvg.device, dtype=torch.float32)
        single = torch.zeros(hvg.shape[0], device=hvg.device, dtype=torch.float32)
        with torch.no_grad():
            real = hvg > 0
            B = hvg.shape[0]
            for b in range(B):
                idx = real[b].nonzero(as_tuple=False).flatten()
                n = int(idx.numel())
                if n == 0:
                    continue
                if self.multi_chunk_dynamic:
                    take = min(int(chunk_len), max(1, math.ceil(n / float(n_chunks))))
                else:
                    take = min(int(chunk_len), n)
                eff_lens[b] = float(take)
                if n <= 1:
                    single[b] = 1.0
                    for c in range(n_chunks):
                        chunks[c][b, idx] = hvg[b, idx]
                    continue
                perm = idx[torch.randperm(n, device=hvg.device)]
                for c in range(n_chunks):
                    start = c * take
                    end = min(start + take, n)
                    keep = perm[start:end] if start < n else idx[torch.randperm(n, device=hvg.device)[:take]]
                    if keep.numel() == 0:
                        keep = idx[torch.randperm(n, device=hvg.device)[:1]]
                    chunks[c][b, keep] = hvg[b, keep]
        return chunks, eff_lens, single

    def _rand_scale_take(self, n: int, scale: tuple[float, float], upper: int, fallback: int) -> int:
        """Sample an I-JEPA-like block size as a fraction of expressed genes."""
        if not self.multi_chunk_dynamic:
            return min(max(1, int(upper)), max(1, n))
        lo, hi = scale
        if hi <= 0:
            return min(max(1, int(fallback)), max(1, n))
        r = torch.empty((), device=self.multi_chunk_target_query.device if self.multi_chunk_target_query is not None else None).uniform_(lo, hi).item()
        take = int(round(max(1.0, n * r)))
        return min(max(1, take), max(1, int(upper)), max(1, n))

    def _sample_context_target_chunk_views(
        self, hvg: torch.Tensor, n_ctx: int, n_tgt: int, chunk_len: int
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample I-JEPA-like context and multiple target gene chunks.

        Target chunks are sampled as multiple random blocks with variable size.
        Context chunks are sampled from the complement of the target union when
        possible, which keeps context/target leakage low while allowing graceful
        fallback for short spots. Target-target overlap is allowed when the spot
        is too short or when random blocks collide.
        """
        ctx_chunks = [torch.zeros_like(hvg) for _ in range(n_ctx)]
        tgt_chunks = [torch.zeros_like(hvg) for _ in range(n_tgt)]
        B = hvg.shape[0]
        ctx_lens = torch.zeros(B, device=hvg.device, dtype=torch.float32)
        tgt_lens = torch.zeros(B, device=hvg.device, dtype=torch.float32)
        single = torch.zeros(B, device=hvg.device, dtype=torch.float32)
        with torch.no_grad():
            real = hvg > 0
            for b in range(B):
                idx = real[b].nonzero(as_tuple=False).flatten()
                n = int(idx.numel())
                if n == 0:
                    continue
                if n <= 1:
                    single[b] = 1.0
                    for ch in ctx_chunks + tgt_chunks:
                        ch[b, idx] = hvg[b, idx]
                    ctx_lens[b] = float(n)
                    tgt_lens[b] = float(n)
                    continue

                target_union = torch.zeros(hvg.shape[1], device=hvg.device, dtype=torch.bool)
                tgt_sizes = []
                for t in range(n_tgt):
                    take = self._rand_scale_take(n, self.multi_chunk_target_scale, chunk_len, max(1, math.ceil(n / float(n_ctx + n_tgt))))
                    available = idx[~target_union[idx]]
                    pool = available if int(available.numel()) >= max(1, min(take, n)) else idx
                    keep = pool[torch.randperm(int(pool.numel()), device=hvg.device)[:min(take, int(pool.numel()))]]
                    if keep.numel() == 0:
                        keep = idx[torch.randperm(n, device=hvg.device)[:1]]
                    tgt_chunks[t][b, keep] = hvg[b, keep]
                    target_union[keep] = True
                    tgt_sizes.append(float(keep.numel()))

                target_free = idx[~target_union[idx]]
                # Context scale is a total fraction; split it across context slots.
                lo, hi = self.multi_chunk_context_scale
                total_ctx_frac = torch.empty((), device=hvg.device).uniform_(lo, hi).item() if self.multi_chunk_dynamic else 1.0
                per_ctx_upper = max(1, int(round(n * total_ctx_frac / float(max(1, n_ctx)))))
                ctx_sizes = []
                used_context = torch.zeros(hvg.shape[1], device=hvg.device, dtype=torch.bool)
                for c in range(n_ctx):
                    pool = target_free[~used_context[target_free]] if target_free.numel() > 0 else idx[~used_context[idx]]
                    if pool.numel() == 0:
                        pool = target_free if target_free.numel() > 0 else idx
                    take = min(int(chunk_len), per_ctx_upper, int(pool.numel()))
                    if take <= 0:
                        take = 1
                    keep = pool[torch.randperm(int(pool.numel()), device=hvg.device)[:take]]
                    if keep.numel() == 0:
                        keep = idx[torch.randperm(n, device=hvg.device)[:1]]
                    ctx_chunks[c][b, keep] = hvg[b, keep]
                    used_context[keep] = True
                    ctx_sizes.append(float(keep.numel()))

                ctx_lens[b] = float(sum(ctx_sizes) / max(1, len(ctx_sizes)))
                tgt_lens[b] = float(sum(tgt_sizes) / max(1, len(tgt_sizes)))
        eff_lens = (ctx_lens + tgt_lens) * 0.5
        return ctx_chunks, tgt_chunks, eff_lens, single, ctx_lens, tgt_lens

    def _target_gene_identity_query(self, hvg_chunk: torch.Tensor) -> torch.Tensor:
        """Return a target-block identity query without leaking expression values.

        I-JEPA gives the predictor target mask positions. For unordered gene
        sets, the closest analogue is the held-out target gene identity set.
        We summarize frozen/current symbol embeddings for the target genes and
        project them to h_tx space. Expression values are not used here.
        """
        B = hvg_chunk.shape[0]
        h_dim = int(getattr(self._student(), "embed_dim", getattr(self._student().tx_encoder, "out_dim", 512)))
        if self.multi_chunk_target_id_proj is None:
            return hvg_chunk.new_zeros((B, h_dim))
        tx = self._student().tx_encoder
        sym = getattr(getattr(getattr(tx, "gene_emb", None), "symbol", None), "emb", None)
        ids = getattr(tx, "hvg_token_ids", None)
        if sym is None or ids is None:
            return hvg_chunk.new_zeros((B, h_dim))
        ids = ids.to(hvg_chunk.device)
        # Detach the descriptor so this query behaves like an I-JEPA position
        # embedding, not a shortcut path updating gene embeddings directly.
        gene_table = sym(ids).detach()  # (K, token_dim)
        mask = (hvg_chunk > 0).float()
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        desc = mask @ gene_table / denom
        return self.multi_chunk_target_id_proj(desc)

    @torch.no_grad()
    def _teacher_target_block_latents(self, full_hvg: torch.Tensor,
                                      tgt_chunks: list[torch.Tensor]) -> list[torch.Tensor]:
        """Extract target-block latents from the full clean EMA teacher pass.

        This mirrors I-JEPA more closely than encoding target chunks in
        isolation: the teacher sees the full spot, then we gather the latent
        tokens corresponding to each target gene block.
        """
        out = self._teacher_forward(full_hvg)
        per = out.get("per_token")
        pos = out.get("orig_positions")
        attn = out.get("attention_mask")
        cls_proj = getattr(self.teacher, "cls_proj", None)
        if per is None or pos is None or attn is None or cls_proj is None:
            targets = []
            for ch in tgt_chunks:
                z = self._teacher_forward(ch)["h_tx"].detach()
                targets.append(F.layer_norm(z, (z.size(-1),)))
            return targets
        pos = pos.clamp(min=0, max=full_hvg.shape[1] - 1)
        attn_b = attn.bool()
        targets = []
        for ch in tgt_chunks:
            tgt_full = ch > 0
            tok_mask = torch.gather(tgt_full, 1, pos) & attn_b
            denom = tok_mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)
            tok_latent = (per * tok_mask.unsqueeze(-1).float()).sum(dim=1) / denom
            z = cls_proj(tok_latent).detach()
            z = F.layer_norm(z, (z.size(-1),))
            targets.append(z)
        return targets

    @torch.no_grad()
    def _teacher_forward(self, hvg: torch.Tensor) -> dict:
        """Run the EMA teacher on the clean, unmasked transcriptomics view."""
        prev_training = self.teacher.training
        prev_force_mask = getattr(self.teacher, "_force_mask_in_eval", False)
        self.teacher.eval()
        if hasattr(self.teacher, "_force_mask_in_eval"):
            self.teacher._force_mask_in_eval = False
        try:
            out = self.teacher(None, hvg, mask=None)
        finally:
            self.teacher.train(prev_training)
            if hasattr(self.teacher, "_force_mask_in_eval"):
                self.teacher._force_mask_in_eval = prev_force_mask
        return out

    @torch.no_grad()
    def _teacher_per_token(self, hvg: torch.Tensor) -> torch.Tensor:
        return self._teacher_forward(hvg)["per_token"]        # (B, K, dim_token)

    # ------------------------------------------------------------------

    def forward(self, model_out: dict, batch: dict) -> tuple[torch.Tensor, dict]:
        has_mask = "tx_mask" in model_out and model_out["tx_mask"] is not None
        mask = model_out.get("tx_mask")
        dev = (mask.device if has_mask else batch["hvg"].device)

        # Multi-chunk / view / DINO / KoLeo paths can be valid even when the
        # encoder emits no masked-token objective, e.g. Stage 1.25 chunk-JEPA
        # refinement with mask_ratio=0.0.  Do not return early just because
        # tx_mask is absent.
        mc_active = self.enable_multi_chunk and self.w_multi_chunk > 0 and "hvg" in batch
        view_active = self.enable_view_jepa and self.w_view_jepa > 0 and "h_tx" in model_out
        dino_active = self.enable_dino and self.w_dino > 0 and "h_tx" in model_out
        koleo_active = self.w_koleo > 0 and "h_tx" in model_out
        if not has_mask and not (mc_active or view_active or dino_active or koleo_active):
            z = torch.zeros((), device=dev)
            return z, {}

        log: dict[str, torch.Tensor] = {}
        loss = torch.zeros((), device=dev)

        student = self._student()
        per_token = model_out.get("tx_per_token")

        # ── Sequence-length tracking ──────────────────────────────────────
        # Tokens in our top_hvg_gene encoder are zero-removed (per-spot
        # variable length).  Log batch stats so we can monitor whether the
        # zero-removal is producing healthy sequences (~150 for HEST,
        # ~30 for spatialcorpus, etc).
        attn = model_out.get("tx_attention_mask")
        if attn is not None:
            with torch.no_grad():
                seq_lens = attn.sum(dim=-1).float()        # (B,) real tokens / spot
                log["tx_self/seq_len_mean"] = seq_lens.mean()
                log["tx_self/seq_len_median"] = seq_lens.median()
                log["tx_self/seq_len_min"] = seq_lens.min()
                log["tx_self/seq_len_max"] = seq_lens.max()
                log["tx_self/seq_len_p10"] = torch.quantile(seq_lens, 0.10)
                log["tx_self/seq_len_p90"] = torch.quantile(seq_lens, 0.90)
                # Explicit aliases: these are encoder-input lengths after
                # vocab clipping, normalization, and max_seq_len sampling.
                log["tx_self/post_sampling_seq_len_mean"] = seq_lens.mean()
                log["tx_self/post_sampling_seq_len_median"] = seq_lens.median()
                log["tx_self/post_sampling_seq_len_min"] = seq_lens.min()
                log["tx_self/post_sampling_seq_len_max"] = seq_lens.max()
                log["tx_self/post_sampling_seq_len_p10"] = torch.quantile(seq_lens, 0.10)
                log["tx_self/post_sampling_seq_len_p90"] = torch.quantile(seq_lens, 0.90)
                pre_lens = batch.get("hvg_seq_len_pre_sampling")
                if pre_lens is not None:
                    pre_lens = pre_lens.to(seq_lens.device).float()
                    log["tx_self/pre_sampling_seq_len_mean"] = pre_lens.mean()
                    log["tx_self/pre_sampling_seq_len_median"] = pre_lens.median()
                    log["tx_self/pre_sampling_seq_len_min"] = pre_lens.min()
                    log["tx_self/pre_sampling_seq_len_max"] = pre_lens.max()
                    log["tx_self/pre_sampling_seq_len_p10"] = torch.quantile(pre_lens, 0.10)
                    log["tx_self/pre_sampling_seq_len_p90"] = torch.quantile(pre_lens, 0.90)
                    log["tx_self/sampling_retention_ratio"] = (seq_lens / pre_lens.clamp(min=1)).mean()
                post_lens = batch.get("hvg_seq_len_post_sampling")
                if post_lens is not None:
                    post_lens = post_lens.to(seq_lens.device).float()
                    log["tx_self/dataset_post_sampling_seq_len_mean"] = post_lens.mean()
                # # masked tokens per spot.  Stage 1.25 chunk-JEPA can run
                # with no token mask, so only log these when a mask exists.
                if has_mask:
                    n_masked = mask.float().sum(dim=-1)
                    log["tx_self/n_masked_mean"] = n_masked.mean()
                    log["tx_self/mask_actual_ratio"] = (n_masked / seq_lens.clamp(min=1)).mean()

        # ── 1. Masked-symbol CE (MSM — primary objective per PDF) ────────
        # Head outputs are computed inside MMAligner.forward (so DDP sees the
        # head params used).  Gated on masking_obj ∈ {"symbol","both"} AND
        # symbol_weight > 0.  Top-k accuracy metrics emitted for monitoring.
        if has_mask and self.use_symbol and self.tx_kind == "top_hvg_gene" \
                and "orig_gene_ids" in model_out \
                and "masked_symbol_logits" in model_out:
            # aligner.forward now emits (N_masked, V) logits — already
            # gathered to masked positions to save 6-7× memory.  Targets are
            # gathered from (B, L) by the same mask.
            logits_m = model_out["masked_symbol_logits"]                  # (N_masked, V)
            tgt = model_out["orig_gene_ids"]                              # (B, L)
            mf = mask
            if mf.any() and logits_m.numel() > 0:
                tgt_m = tgt[mf]
                l_sym = F.cross_entropy(logits_m, tgt_m)
                loss = loss + self.w_symbol * l_sym
                log["tx_self/masked_symbol_ce"] = l_sym.detach()
                vocab_size = int(logits_m.size(-1))
                random_ce = math.log(max(vocab_size, 2))
                log["tx_self/masked_symbol_vocab_size"] = torch.as_tensor(
                    float(vocab_size), device=logits_m.device
                )
                log["tx_self/masked_symbol_random_ce"] = torch.as_tensor(
                    random_ce, device=logits_m.device
                )
                log["tx_self/masked_symbol_ce_norm"] = (l_sym / max(random_ce, 1e-8)).detach()
                log["tx_self/masked_symbol_ce_gain"] = torch.as_tensor(
                    random_ce, device=logits_m.device
                ) - l_sym.detach()
                # Top-k accuracy (k = 1, 5, 10) for monitoring.
                with torch.no_grad():
                    for k in (1, 5, 10):
                        if logits_m.size(-1) >= k:
                            topk = logits_m.topk(k, dim=-1).indices       # (N, k)
                            hit = (topk == tgt_m.unsqueeze(-1)).any(-1).float().mean()
                            log[f"tx_self/masked_symbol_top{k}_acc"] = hit
                    # alias for back-compat
                    log["tx_self/masked_symbol_acc"] = log["tx_self/masked_symbol_top1_acc"]

        # ── 2. Masked-value MSE ───────────────────────────────────────────
        # Gated on masking_obj ∈ {"value","both"} AND value_weight > 0.
        # Gene-wise PCC also logged when path active.
        if has_mask and self.use_value:
            pred_m = None
            if "masked_value_pred" in model_out and self.tx_kind == "top_hvg_gene":
                # aligner emits (N_masked,) for top_hvg_gene — already gathered.
                pred_m = model_out["masked_value_pred"]
            elif "masked_value_pred" in model_out:
                pred_m = model_out["masked_value_pred"][mask.bool()]      # (N_masked,)
            elif self.tx_kind == "hvg_tokenizer" and hasattr(student.tx_encoder, "encoder") \
                    and hasattr(student.tx_encoder.encoder, "recon_head"):
                pred_full = student.tx_encoder.encoder.recon_head(per_token).squeeze(-1)
                pred_m = pred_full[mask.bool()]
            if pred_m is not None and pred_m.numel() > 0:
                if self.tx_kind == "top_hvg_gene" and "orig_positions" in model_out:
                    tgt_full = torch.gather(batch["hvg"], 1, model_out["orig_positions"])
                else:
                    tgt_full = batch["hvg"]
                tgt_m = tgt_full[mask.bool()]
                l_val = ((pred_m - tgt_m).pow(2)).mean()
                loss = loss + self.w_value * l_val
                log["tx_self/masked_value_mse"] = l_val.detach()
                # Per-token Pearson — aggregated over masked positions.
                with torch.no_grad():
                    if pred_m.numel() > 1:
                        p_m = pred_m.float(); t_m = tgt_m.float()
                        p_c = p_m - p_m.mean(); t_c = t_m - t_m.mean()
                        denom_pcc = (p_c.std() * t_c.std() * p_m.numel()).clamp(min=1e-8)
                        pcc = (p_c * t_c).sum() / denom_pcc
                        log["tx_self/masked_value_pcc"] = pcc.detach()
                        # Rank correlation is a better monitor when gene_norm
                        # encodes relative salience (e.g. global_median).
                        p_rank = torch.argsort(torch.argsort(p_m))
                        t_rank = torch.argsort(torch.argsort(t_m))
                        pr = p_rank.float() - p_rank.float().mean()
                        tr = t_rank.float() - t_rank.float().mean()
                        denom_sp = (pr.std() * tr.std() * pr.numel()).clamp(min=1e-8)
                        log["tx_self/masked_value_spearman"] = ((pr * tr).sum() / denom_sp).detach()

        # ── 3. Masked-JEPA (optional) ─────────────────────────────
        # Per SPATULA paper (page 14): predict masked-token latent via predictor;
        # target = EMA teacher with stop-grad; loss = smooth-L1 in latent space.
        # (Previous version used cosine which is a JEPA variant but not what the
        # paper specifies.)
        if has_mask and self.enable_jepa:
            pred_m = None
            if "masked_jepa_pred" in model_out and self.tx_kind == "top_hvg_gene":
                # aligner emits (N_masked, D) already.
                pred_m = model_out["masked_jepa_pred"]
            elif "masked_jepa_pred" in model_out:
                pred_m = model_out["masked_jepa_pred"][mask.bool()]
            elif self.tx_kind == "hvg_tokenizer" and hasattr(student.tx_encoder, "encoder") \
                    and hasattr(student.tx_encoder.encoder, "jepa_head"):
                pred_full = student.tx_encoder.encoder.jepa_head(per_token)
                pred_m = pred_full[mask.bool()]
            if pred_m is not None and pred_m.numel() > 0:
                with torch.no_grad():
                    teacher_tok = self._teacher_per_token(batch["hvg"])    # (B, L, D)
                tgt_m = teacher_tok[mask.bool()].detach()                  # (N_masked, D)
                per_elem = F.smooth_l1_loss(pred_m, tgt_m, reduction="none", beta=1.0)
                l_jepa = per_elem.mean()
                loss = loss + self.w_jepa * l_jepa
                log["tx_self/masked_jepa_smoothl1"] = l_jepa.detach()
                with torch.no_grad():
                    tn = F.normalize(tgt_m, dim=-1)
                    pn = F.normalize(pred_m, dim=-1)
                    log["tx_self/masked_jepa_cosine"] = (2 - 2 * (pn * tn).sum(-1)).mean()

        # ── 4. View-JEPA / data2vec-style spot consistency (optional) ─
        # Teacher view: clean, unmasked expression sequence.
        # Student view: the current masked/noisy encoder pass.
        # Unlike direct DINO matching, a predictor absorbs view-specific
        # corruption so h_tx itself is less forced to become invariant too early.
        view_scale = self._warmup_scale(self.view_jepa_warmup_epochs, self.view_jepa_ramp_epochs)
        view_weight_eff = self.w_view_jepa * view_scale
        if self.enable_view_jepa and self.w_view_jepa > 0 and "h_tx" in model_out:
            log["tx_self/view_jepa_weight_effective"] = torch.as_tensor(
                float(view_weight_eff), device=model_out["h_tx"].device
            )
            log["tx_self/view_jepa_warmup_scale"] = torch.as_tensor(
                float(view_scale), device=model_out["h_tx"].device
            )
        if self.enable_view_jepa and self.view_jepa_predictor is not None and view_weight_eff > 0 and "h_tx" in model_out:
            with torch.no_grad():
                teacher_out = self._teacher_forward(batch["hvg"])
                tgt = teacher_out["h_tx"].detach()
            pred = self.view_jepa_predictor(model_out["h_tx"])
            if pred.shape == tgt.shape and pred.numel() > 0:
                if self.view_jepa_loss == "cosine":
                    pn = F.normalize(pred, dim=-1)
                    tn = F.normalize(tgt - tgt.mean(dim=0, keepdim=True).detach(), dim=-1)
                    l_view = (2 - 2 * (pn * tn).sum(-1)).mean()
                else:
                    l_view = F.smooth_l1_loss(pred, tgt, beta=1.0)
                loss = loss + view_weight_eff * l_view
                log["tx_self/view_jepa"] = l_view.detach()
                with torch.no_grad():
                    pn = F.normalize(pred, dim=-1)
                    tn = F.normalize(tgt, dim=-1)
                    log["tx_self/view_jepa_cosine_distance"] = (2 - 2 * (pn * tn).sum(-1)).mean()

        # ── 5. Multi-chunk Spot-JEPA (optional) ────────────────────
        # `batch["hvg"]` is the non-zero sequence carrier. We sample context
        # and target chunks from it. The JEPA target is target chunk latent(s),
        # I-JEPA-style; z_spot is reserved for inference/eval pooling.
        mc_scale = self._warmup_scale(self.multi_chunk_warmup_epochs, self.multi_chunk_ramp_epochs)
        mc_weight_eff = self.w_multi_chunk * mc_scale
        if self.enable_multi_chunk and self.w_multi_chunk > 0 and "hvg" in batch:
            dev = batch["hvg"].device
            log["tx_self/multi_chunk_weight_effective"] = torch.as_tensor(float(mc_weight_eff), device=dev)
            log["tx_self/multi_chunk_warmup_scale"] = torch.as_tensor(float(mc_scale), device=dev)
            log["tx_self/multi_chunk_n_chunks"] = torch.as_tensor(float(self.multi_chunk_n), device=dev)
            log["tx_self/multi_chunk_len"] = torch.as_tensor(float(self.multi_chunk_len), device=dev)
            log["tx_self/multi_chunk_dynamic"] = torch.as_tensor(float(self.multi_chunk_dynamic), device=dev)
            log["tx_self/multi_chunk_target_chunks"] = torch.as_tensor(float(self.multi_chunk_target_chunks), device=dev)
            log["tx_self/multi_chunk_target_chunks_auto"] = torch.as_tensor(float(self.multi_chunk_target_chunks_auto), device=dev)
            log["tx_self/multi_chunk_target_scale_min"] = torch.as_tensor(float(self.multi_chunk_target_scale[0]), device=dev)
            log["tx_self/multi_chunk_target_scale_max"] = torch.as_tensor(float(self.multi_chunk_target_scale[1]), device=dev)
            log["tx_self/multi_chunk_context_scale_min"] = torch.as_tensor(float(self.multi_chunk_context_scale[0]), device=dev)
            log["tx_self/multi_chunk_context_scale_max"] = torch.as_tensor(float(self.multi_chunk_context_scale[1]), device=dev)
        if self.enable_multi_chunk and self.multi_chunk_predictor is not None and mc_weight_eff > 0 and "hvg" in batch:
            n_tgt = min(self.multi_chunk_target_chunks, max(1, self.multi_chunk_n - 1))
            n_ctx = max(1, self.multi_chunk_n - n_tgt)
            ctx_chunks, tgt_chunks, eff_lens, single, ctx_lens, tgt_lens = self._sample_context_target_chunk_views(
                batch["hvg"], n_ctx, n_tgt, self.multi_chunk_len
            )
            chunks = ctx_chunks + tgt_chunks
            ctx_z = [self._student_forward_clean(ch)["h_tx"] for ch in ctx_chunks]
            z_context = torch.stack(ctx_z, dim=0).mean(dim=0)

            per_target_losses = []
            per_target_cosine = []
            context_target_losses = []
            context_target_cosine = []
            query_only_losses = []
            if self.multi_chunk_target == "spot_aggregate":
                # Kept only as an ablation option. Main JEPA uses target_chunk.
                with torch.no_grad():
                    tgt_z = [self._teacher_forward(ch)["h_tx"].detach() for ch in chunks]
                    z_tgt = torch.stack(tgt_z, dim=0).mean(dim=0)
                pred = self.multi_chunk_predictor(z_context)
                target_pairs = [(pred, z_tgt)]
            else:
                # I-JEPA-style multi-block target: predict each target chunk
                # separately, then average the per-target losses. We do not
                # average target chunks before prediction.
                with torch.no_grad():
                    tgt_z = self._teacher_target_block_latents(batch["hvg"], tgt_chunks)
                target_pairs = []
                target_id_norms = []
                target_id_raw_norms = []
                for i, z_tgt in enumerate(tgt_z):
                    pred_in = z_context
                    query_only_in = torch.zeros_like(z_context)
                    if self.multi_chunk_target_query is not None:
                        q_idx = min(n_ctx + i, self.multi_chunk_target_query.shape[0] - 1)
                        slot_q = self.multi_chunk_target_query[q_idx].unsqueeze(0)
                        pred_in = pred_in + slot_q
                        query_only_in = query_only_in + slot_q
                    target_id_query_raw = self._target_gene_identity_query(tgt_chunks[i])
                    target_id_raw_norms.append(target_id_query_raw.norm(dim=-1).mean().detach())
                    target_id_query = self.multi_chunk_target_id_scale * target_id_query_raw
                    target_id_norms.append(target_id_query.norm(dim=-1).mean().detach())
                    pred_in = pred_in + target_id_query
                    query_only_in = query_only_in + target_id_query
                    target_pairs.append((self.multi_chunk_predictor(pred_in), z_tgt))
                    with torch.no_grad():
                        pred_query_only = self.multi_chunk_predictor(query_only_in)
                        query_only_losses.append(F.smooth_l1_loss(pred_query_only, z_tgt, beta=1.0))

            for pred, z_tgt in target_pairs:
                if self.multi_chunk_loss == "cosine":
                    pn = F.normalize(pred, dim=-1)
                    tn = F.normalize(z_tgt, dim=-1)
                    l_tgt = (2 - 2 * (pn * tn).sum(-1)).mean()
                else:
                    l_tgt = F.smooth_l1_loss(pred, z_tgt, beta=1.0)
                per_target_losses.append(l_tgt)
                with torch.no_grad():
                    pn = F.normalize(pred, dim=-1)
                    cn = F.normalize(z_context, dim=-1)
                    tn = F.normalize(z_tgt, dim=-1)
                    per_target_cosine.append((2 - 2 * (pn * tn).sum(-1)).mean())
                    context_target_cosine.append((2 - 2 * (cn * tn).sum(-1)).mean())
                    context_target_losses.append(F.smooth_l1_loss(z_context, z_tgt, beta=1.0))

            l_mc = torch.stack(per_target_losses).mean()
            weighted_mc = mc_weight_eff * l_mc
            loss = loss + weighted_mc
            z_reg = torch.cat([z_context] + [pred for pred, _ in target_pairs], dim=0)
            log["tx_self/multi_chunk_regularizer"] = torch.as_tensor(
                {"none": 0.0, "koleo": 1.0, "vicreg": 2.0}[self.multi_chunk_regularizer],
                device=z_context.device,
            )
            if self.multi_chunk_regularizer == "koleo":
                mc_koleo_weight_eff = self.w_multi_chunk_koleo * mc_scale
                if mc_koleo_weight_eff > 0:
                    l_mc_koleo = _koleo_loss(z_reg, eps=self.koleo_eps)
                    loss = loss + mc_koleo_weight_eff * l_mc_koleo
                    log["tx_self/multi_chunk_koleo"] = l_mc_koleo.detach()
                    log["tx_self/multi_chunk_koleo_weighted"] = (mc_koleo_weight_eff * l_mc_koleo).detach()
                    log["tx_self/multi_chunk_koleo_weight_effective"] = torch.as_tensor(
                        float(mc_koleo_weight_eff), device=z_context.device
                    )
            elif self.multi_chunk_regularizer == "vicreg":
                l_var, l_cov = _vicreg_var_cov_loss(
                    z_reg, gamma=self.multi_chunk_vicreg_gamma, eps=1e-4
                )
                reg_weight_eff = mc_scale
                l_vicreg = self.w_multi_chunk_vicreg_var * l_var + self.w_multi_chunk_vicreg_cov * l_cov
                loss = loss + reg_weight_eff * l_vicreg
                log["tx_self/multi_chunk_vicreg_var"] = l_var.detach()
                log["tx_self/multi_chunk_vicreg_cov"] = l_cov.detach()
                log["tx_self/multi_chunk_vicreg"] = l_vicreg.detach()
                log["tx_self/multi_chunk_vicreg_weighted"] = (reg_weight_eff * l_vicreg).detach()
                log["tx_self/multi_chunk_vicreg_weight_effective"] = torch.as_tensor(
                    float(reg_weight_eff), device=z_context.device
                )
                log["tx_self/multi_chunk_vicreg_var_weight"] = torch.as_tensor(
                    float(self.w_multi_chunk_vicreg_var), device=z_context.device
                )
                log["tx_self/multi_chunk_vicreg_cov_weight"] = torch.as_tensor(
                    float(self.w_multi_chunk_vicreg_cov), device=z_context.device
                )
            log["tx_self/multi_chunk_jepa"] = l_mc.detach()
            log["tx_self/multi_chunk_jepa_weighted"] = weighted_mc.detach()
            if "tx_self/masked_symbol_ce" in log:
                log["tx_self/multi_chunk_jepa_to_msm_ratio"] = (
                    weighted_mc.detach() / log["tx_self/masked_symbol_ce"].clamp(min=1e-8)
                )
            with torch.no_grad():
                log["tx_self/multi_chunk_cosine_distance"] = torch.stack(per_target_cosine).mean()
                log["tx_self/multi_chunk_context_target_cosine_distance"] = torch.stack(context_target_cosine).mean()
                log["tx_self/multi_chunk_context_target_smoothl1"] = torch.stack(context_target_losses).mean()
                # Positive means the predictor improves over raw context -> target matching.
                log["tx_self/multi_chunk_predictor_smoothl1_gain"] = (
                    log["tx_self/multi_chunk_context_target_smoothl1"] - l_mc.detach()
                )
                log["tx_self/multi_chunk_target_is_spot"] = torch.as_tensor(
                    float(self.multi_chunk_target == "spot_aggregate"), device=pred.device
                )
                log["tx_self/multi_chunk_n_targets"] = torch.as_tensor(
                    float(len(target_pairs)), device=pred.device
                )
                if "target_id_norms" in locals() and target_id_norms:
                    log["tx_self/multi_chunk_target_id_query_norm"] = torch.stack(target_id_norms).mean()
                    log["tx_self/multi_chunk_target_id_query_raw_norm"] = torch.stack(target_id_raw_norms).mean()
                    log["tx_self/multi_chunk_target_id_scale"] = torch.as_tensor(
                        float(self.multi_chunk_target_id_scale), device=pred.device
                    )
                if query_only_losses:
                    q_loss = torch.stack(query_only_losses).mean()
                    log["tx_self/multi_chunk_query_only_smoothl1"] = q_loss
                    # Positive means context+gene-list beats gene-list/slot query alone.
                    log["tx_self/multi_chunk_context_gain_over_query_only"] = (q_loss - l_mc.detach())
                log["tx_self/multi_chunk_eff_len_mean"] = eff_lens.mean()
                log["tx_self/multi_chunk_context_eff_len_mean"] = ctx_lens.mean()
                log["tx_self/multi_chunk_target_eff_len_mean"] = tgt_lens.mean()
                log["tx_self/multi_chunk_single_chunk_frac"] = single.mean()
                ctx_stack = torch.stack([(ch > 0) for ch in ctx_chunks], dim=0)
                tgt_stack = torch.stack([(ch > 0) for ch in tgt_chunks], dim=0)
                ctx_mask = ctx_stack.any(dim=0)
                tgt_mask = tgt_stack.any(dim=0)
                real_mask = batch["hvg"] > 0
                real_size = real_mask.float().sum(dim=1).clamp(min=1.0)
                overlap = (ctx_mask & tgt_mask).float().sum(dim=1)
                target_size = tgt_mask.float().sum(dim=1).clamp(min=1.0)
                context_size = ctx_mask.float().sum(dim=1).clamp(min=1.0)
                log["tx_self/multi_chunk_ctx_tgt_overlap"] = (overlap / target_size).mean()
                log["tx_self/multi_chunk_context_union_frac"] = (context_size / real_size).mean()
                log["tx_self/multi_chunk_target_union_frac"] = (target_size / real_size).mean()
                log["tx_self/multi_chunk_residual_after_context_frac"] = (
                    ((real_mask & ~ctx_mask).float().sum(dim=1) / real_size).mean()
                )
                if tgt_stack.shape[0] >= 2:
                    pair_vals = []
                    for i_t in range(tgt_stack.shape[0]):
                        for j_t in range(i_t + 1, tgt_stack.shape[0]):
                            inter = (tgt_stack[i_t] & tgt_stack[j_t]).float().sum(dim=1)
                            denom = (tgt_stack[i_t] | tgt_stack[j_t]).float().sum(dim=1).clamp(min=1.0)
                            pair_vals.append(inter / denom)
                    if pair_vals:
                        log["tx_self/multi_chunk_target_target_jaccard"] = torch.stack(pair_vals, dim=0).mean()
                if len(ctx_z) >= 2:
                    z0 = F.normalize(ctx_z[0], dim=-1)
                    z1 = F.normalize(ctx_z[1], dim=-1)
                    log["tx_self/multi_chunk_context_cosine"] = (z0 * z1).sum(-1).mean()

        # ── 6. DINO-style spot consistency (optional) ─────────────
        # Teacher view: clean, unmasked expression sequence.
        # Student view: the current masked/noisy encoder pass.
        # MSM keeps this consistency term tied to gene-token semantics.
        dino_scale = self._warmup_scale(self.dino_warmup_epochs, self.dino_ramp_epochs)
        dino_weight_eff = self.w_dino * dino_scale
        if self.enable_dino and self.w_dino > 0 and "h_tx" in model_out:
            log["tx_self/dino_weight_effective"] = torch.as_tensor(
                float(dino_weight_eff), device=model_out["h_tx"].device
            )
            log["tx_self/dino_warmup_scale"] = torch.as_tensor(
                float(dino_scale), device=model_out["h_tx"].device
            )
        if self.enable_dino and dino_weight_eff > 0 and "h_tx" in model_out:
            with torch.no_grad():
                teacher_out = self._teacher_forward(batch["hvg"])
                tgt = teacher_out["h_tx"].detach()
            pred = model_out["h_tx"]
            if pred.shape == tgt.shape and pred.numel() > 0:
                pn = F.normalize(pred, dim=-1)
                tgt_centered = tgt - tgt.mean(dim=0, keepdim=True).detach()
                tn = F.normalize(tgt_centered, dim=-1)
                cos_dist = (2 - 2 * (pn * tn).sum(-1)).mean()
                if self.dino_loss == "smooth_l1":
                    l_dino = F.smooth_l1_loss(pred, tgt, beta=1.0)
                else:                                   # cosine (default)
                    l_dino = cos_dist
                loss = loss + dino_weight_eff * l_dino
                log["tx_self/dino_consistency"] = l_dino.detach()
                log["tx_self/dino_cosine_distance"] = cos_dist.detach()

        # ── 6. KoLeo embedding entropy regularizer (optional) ───────
        koleo_scale = self._warmup_scale(self.koleo_warmup_epochs, self.koleo_ramp_epochs)
        koleo_weight_eff = self.w_koleo * koleo_scale
        if self.w_koleo > 0 and "h_tx" in model_out:
            log["tx_self/koleo_weight_effective"] = torch.as_tensor(
                float(koleo_weight_eff), device=model_out["h_tx"].device
            )
            log["tx_self/koleo_warmup_scale"] = torch.as_tensor(
                float(koleo_scale), device=model_out["h_tx"].device
            )
        if koleo_weight_eff > 0 and "h_tx" in model_out:
            l_koleo = _koleo_loss(model_out["h_tx"], eps=self.koleo_eps)
            loss = loss + koleo_weight_eff * l_koleo
            log["tx_self/koleo"] = l_koleo.detach()

        log["tx_self/loss"] = loss.detach()
        return loss, log

    def on_after_step(self) -> None:
        if self.teacher is not None:
            _ema_update(self._student().tx_encoder, self.teacher, self.m)
