"""Render a method-overview figure for mm_align.

Sections:
  1. Data preparation (H&E patches + Visium ST → paired shards)
  2. Multimodal alignment architecture (UNI image encoder + Novae+HVG tx encoder
     → shared latent → projectors)
  3. Five alignment objectives (CLIP / VICReg / JEPA / CrossAttn / MMAE) plus
     Spatial-JEPA and masked-gene-modeling auxiliary
  4. Evaluation (zero-shot vs MLP linear probe)

Output: results/figures/method.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D


# ----- palette -----
C_IMG = "#cfe4ff"          # image side (light blue)
C_TX  = "#ffe2b8"          # tx side (light orange)
C_SHR = "#d4f0d4"          # shared latent (light green)
C_OBJ = "#f3d4ff"          # objectives (light purple)
C_EVAL = "#fcd9d9"         # eval (light pink)
C_DATA = "#eaeaea"         # data prep (light grey)
C_EDGE = "#444444"
C_ACC = "#1f3c88"          # accent line


def box(ax, xy, wh, text, color, *, fontsize=9, edge=C_EDGE, lw=1.2, bold=False,
        text_color="#1a1a1a", pad=0.02):
    x, y = xy; w, h = wh
    patch = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad={pad},rounding_size=0.06",
                           linewidth=lw, edgecolor=edge, facecolor=color)
    ax.add_patch(patch)
    weight = "bold" if bold else "normal"
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, weight=weight, color=text_color)
    return (x, y, w, h)


def section_title(ax, x, y, label, color):
    ax.text(x, y, label, fontsize=12, weight="bold", color=color,
            ha="left", va="center")


def arrow(ax, src, dst, *, color=C_EDGE, lw=1.4, style="->", connectionstyle="arc3,rad=0",
          shrinkA=4, shrinkB=4):
    a = FancyArrowPatch(src, dst, arrowstyle=style, color=color, lw=lw,
                        connectionstyle=connectionstyle,
                        shrinkA=shrinkA, shrinkB=shrinkB,
                        mutation_scale=12)
    ax.add_patch(a)


def center(rect):
    x, y, w, h = rect
    return (x + w / 2, y + h / 2)


def top(rect):
    x, y, w, h = rect
    return (x + w / 2, y + h)


def bottom(rect):
    x, y, w, h = rect
    return (x + w / 2, y)


def left(rect):
    x, y, w, h = rect
    return (x, y + h / 2)


def right(rect):
    x, y, w, h = rect
    return (x + w, y + h / 2)


def main():
    out = Path("results/figures/method.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 11))
    ax.set_xlim(0, 14); ax.set_ylim(0, 11)
    ax.set_aspect("equal"); ax.axis("off")

    # =====================================================================
    # Title
    # =====================================================================
    ax.text(7, 10.55, "mm_align — Pathology × Spatial-Transcriptomics alignment on HEST-1k",
            ha="center", va="center", fontsize=16, weight="bold")
    ax.text(7, 10.18, "Per-spot multimodal alignment with frozen Novae transcript encoder and tunable UNI image encoder",
            ha="center", va="center", fontsize=10, color="#555555", style="italic")

    # =====================================================================
    # SECTION 1 — Data preparation
    # =====================================================================
    section_title(ax, 0.3, 9.55, "① Data preparation", C_ACC)
    ax.add_patch(Rectangle((0.2, 7.6), 13.6, 1.95, fill=False, edgecolor="#bbbbbb", lw=0.8, linestyle=":"))

    # H&E WSI
    he = box(ax, (0.6, 8.45), (1.7, 0.55), "H&E WSI\n(/data/hest/wsis)", C_DATA, fontsize=8)
    patch_h5 = box(ax, (0.6, 7.75), (1.7, 0.55),
                   "Patches H5\n224×224 uint8 + UNI 1536", C_IMG, fontsize=8)
    arrow(ax, bottom(he), top(patch_h5))

    # Visium adata
    adata = box(ax, (3.0, 8.45), (1.7, 0.55), "Visium .h5ad\n(/data/hest/st)", C_DATA, fontsize=8)
    novae_box = box(ax, (3.0, 7.75), (1.7, 0.55),
                    "Novae (frozen)\n→ 64-d latent  +  HVG 2048", C_TX, fontsize=8)
    arrow(ax, bottom(adata), top(novae_box))

    # Whitelist
    wl = box(ax, (5.4, 8.45), (1.7, 0.55), "human_list.txt\n(649 Homo sapiens)", C_DATA, fontsize=8)
    arrow(ax, right(wl), (8.0, 8.72), connectionstyle="arc3,rad=0.15")

    # Paired shard
    shard = box(ax, (8.0, 7.85), (4.5, 1.20),
                "Paired shard  results/cache/prepared/{sid}.h5\n"
                "barcode • coords • uni_feat[1536] • patch_idx\n"
                "novae_latent[64] • hvg_log[2048]",
                C_SHR, fontsize=9, bold=True)
    arrow(ax, right(patch_h5), (8.0, 8.40), connectionstyle="arc3,rad=-0.12")
    arrow(ax, right(novae_box), (8.0, 8.20), connectionstyle="arc3,rad=-0.06")

    # =====================================================================
    # SECTION 2 — Model architecture
    # =====================================================================
    section_title(ax, 0.3, 7.20, "② Multimodal alignment model (per spot)", C_ACC)
    ax.add_patch(Rectangle((0.2, 4.20), 13.6, 3.0, fill=False, edgecolor="#bbbbbb", lw=0.8, linestyle=":"))

    # Image side
    img_raw = box(ax, (0.6, 6.45), (3.1, 0.55),
                  "224×224 patch", C_IMG, fontsize=9)
    uni = box(ax, (0.6, 5.55), (3.1, 0.7),
              "UNI ViT-G/14  (681M)\nfreeze:  feature | all | partial:N | none",
              C_IMG, fontsize=8, bold=True)
    img_adapt = box(ax, (0.9, 4.80), (2.5, 0.45),
                    "Image adapter MLP\n→ h_image (256)", C_IMG, fontsize=8)
    arrow(ax, bottom(img_raw), top(uni))
    arrow(ax, bottom(uni), top(img_adapt))

    # Tx side
    tx_in = box(ax, (4.7, 6.45), (3.1, 0.55),
                "novae 64 + HVG 2048", C_TX, fontsize=9)
    tx_enc = box(ax, (4.7, 5.55), (3.1, 0.7),
                 "TranscriptEncoder MLP\n(adapter; novae model itself frozen)",
                 C_TX, fontsize=8, bold=True)
    tx_h = box(ax, (5.0, 4.80), (2.5, 0.45),
               "→ h_tx (256)", C_TX, fontsize=8)
    arrow(ax, bottom(tx_in), top(tx_enc))
    arrow(ax, bottom(tx_enc), top(tx_h))

    # Shared projector
    proj_i = box(ax, (8.5, 5.65), (1.7, 0.55), "Image projector\nMLP→256", C_SHR, fontsize=8)
    proj_t = box(ax, (8.5, 4.85), (1.7, 0.55), "Tx projector\nMLP→256", C_SHR, fontsize=8)
    arrow(ax, right(img_adapt), left(proj_i), connectionstyle="arc3,rad=-0.15")
    arrow(ax, right(tx_h), left(proj_t), connectionstyle="arc3,rad=0.15")

    # Latents
    z_i = box(ax, (11.0, 5.65), (2.4, 0.55), "z_image (256)", C_SHR, fontsize=9, bold=True)
    z_t = box(ax, (11.0, 4.85), (2.4, 0.55), "z_tx (256)",    C_SHR, fontsize=9, bold=True)
    arrow(ax, right(proj_i), left(z_i))
    arrow(ax, right(proj_t), left(z_t))

    # h_image / h_tx → downstream
    h_label = box(ax, (3.85, 4.30), (4.3, 0.30),
                  "(h_image, h_tx) used downstream for retrieval / probe / UMAP",
                  "#ffffff", fontsize=8)

    # =====================================================================
    # SECTION 3 — Objectives
    # =====================================================================
    section_title(ax, 0.3, 3.85, "③ Alignment objectives (one per training run)", C_ACC)
    ax.add_patch(Rectangle((0.2, 2.55), 13.6, 1.3, fill=False, edgecolor="#bbbbbb", lw=0.8, linestyle=":"))

    obj_w = 2.35; obj_y = 2.95; obj_h = 0.6
    titles = ["CLIP", "VICReg", "JEPA", "Cross-attention", "MMAE"]
    subs = [
        "InfoNCE on (z_i, z_t)",
        "inv + var + cov terms",
        "EMA teacher + predictor\n+ Spatial-JEPA on KNN",
        "image-attn-tx & tx-attn-image\nthen contrastive",
        "mask HVG, reconstruct\nfrom h_image+visible-tx",
    ]
    for k, (t, s) in enumerate(zip(titles, subs)):
        bx = 0.4 + k * (obj_w + 0.20)
        box(ax, (bx, obj_y), (obj_w, obj_h), f"{t}\n{s}", C_OBJ, fontsize=8, bold=False)

    # auxiliary
    box(ax, (0.4, 2.60), (6.5, 0.30),
        "AUX:  masked gene-token reconstruction (linear gene head)",
        C_OBJ, fontsize=8)
    box(ax, (7.1, 2.60), (6.4, 0.30),
        "AUX:  Spatial-JEPA — predict spot from K spatial neighbours",
        C_OBJ, fontsize=8)

    # arrows from latents to objective row
    arrow(ax, bottom(z_i), (7.0, 3.85), connectionstyle="arc3,rad=-0.05", color=C_ACC)
    arrow(ax, bottom(z_t), (7.0, 3.85), connectionstyle="arc3,rad=0.05",  color=C_ACC)

    # =====================================================================
    # SECTION 4 — Evaluation
    # =====================================================================
    section_title(ax, 0.3, 2.25, "④ Evaluation", C_ACC)
    ax.add_patch(Rectangle((0.2, 0.20), 13.6, 2.05, fill=False, edgecolor="#bbbbbb", lw=0.8, linestyle=":"))

    # Zero-shot block
    box(ax, (0.4, 0.40), (6.45, 1.7),
        "ZERO-SHOT (every epoch, no training)\n\n"
        "• retrieval i2t / t2i  R@K, MRR\n"
        "• clustering silhouette + ARI/NMI vs organ\n"
        "• RankMe (effective rank), modality gap\n"
        "• alignment / uniformity (Wang & Isola)\n"
        "• UMAP panel  →  umap_latest.png",
        C_EVAL, fontsize=8.5, bold=False, text_color="#1a1a1a")
    ax.text(0.55, 1.90, "  scripts/eval/zero_shot.py", fontsize=8, style="italic", color="#555")

    # Linear probe block
    box(ax, (7.15, 0.40), (6.4, 1.7),
        "MLP LINEAR PROBE (end of training; per-epoch optional)\n\n"
        "• HVG regression: image-side → 2048-d log1p\n"
        "    mse, mean Pearson, top-50 Pearson\n"
        "• Organ classification: image-side → organ class\n"
        "    accuracy, macro-F1\n"
        "Arms: ours_image_only / ours_multimodal / baseline_uni",
        C_EVAL, fontsize=8.5, bold=False, text_color="#1a1a1a")
    ax.text(7.30, 1.90, "  scripts/eval/linear_probe.py", fontsize=8, style="italic", color="#555")

    # Legend
    handles = [
        Line2D([0], [0], marker="s", linestyle="", markersize=12, markerfacecolor=C_IMG,
               markeredgecolor=C_EDGE, label="Image stream"),
        Line2D([0], [0], marker="s", linestyle="", markersize=12, markerfacecolor=C_TX,
               markeredgecolor=C_EDGE, label="Transcriptomics stream"),
        Line2D([0], [0], marker="s", linestyle="", markersize=12, markerfacecolor=C_SHR,
               markeredgecolor=C_EDGE, label="Shared latent / paired data"),
        Line2D([0], [0], marker="s", linestyle="", markersize=12, markerfacecolor=C_OBJ,
               markeredgecolor=C_EDGE, label="Objective"),
        Line2D([0], [0], marker="s", linestyle="", markersize=12, markerfacecolor=C_EVAL,
               markeredgecolor=C_EDGE, label="Evaluation"),
    ]
    ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.995, 0.995),
              fontsize=8, frameon=True, ncol=1)

    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
