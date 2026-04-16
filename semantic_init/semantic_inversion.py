"""
semantic_inversion.py — Semantic warm-start feature inversion for Qwen VL.

Hypothesis
----------
Gradient-based inversion starting from gray noise tends to converge to blurry
local minima. Starting from a semantically-similar anchor image (chosen from a
small curated library via cosine-similarity matching in feature space) should
give better reconstruction, because:
  • the optimization starts closer to the target in feature space;
  • TV and L2 penalties are gentler on natural images than on random noise;
  • the gradient landscape is smoother near a real image.

Algorithm
---------
1. Load an AnchorLibrary from a directory (one image per category, e.g.
   animal.jpg, building.jpg, ...).
2. For each target layer, encode all anchors at that layer and select the one
   with the highest cosine similarity to the target features (mean-pooled over
   the token dimension).
3. Warm-start: --method best | mixup | knn-mixup (switch only this between runs).
4. Optionally also run a gray-init baseline (--compare) to measure the gain.

TODO (not implemented yet): distance-weighted mixup; softmax weights; multiple
images per category in anchors/.

Usage
-----
    If the repo (or any ancestor directory) contains models/hub/ with a HF hub
    cache for Qwen/Qwen3.5-4B, HF_HUB_CACHE is set automatically so weights load
    from disk.

    # First time: set up anchor images
    python prepare_anchors.py --anchors-dir anchors/

    # Default output: results/runs/<timestamp>/; add --compare to run gray baseline
    python semantic_inversion.py \\
        --model-name Qwen/Qwen3.5-4B \\
        --image /path/to/target.jpg \\
        --anchors-dir anchors/ \\
        --layers 1,4,8,16,last \\
        --method knn-mixup \\
        --steps 1500 \\
        --restarts 3
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

# ── shared helpers from parent directory ─────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))
def _resolve_output_dir(
    output_dir_arg: str | None,
    script_dir: Path,
) -> tuple[Path, str]:
    """
    If output_dir_arg is set, resolve under script_dir when relative.
    Otherwise results/runs/<YYYYMMDD-HHMMSS>/ to avoid overwriting past runs.
    """
    if output_dir_arg:
        p = Path(output_dir_arg)
        if not p.is_absolute():
            p = script_dir / p
        return p, p.name
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    p = script_dir / "results" / "runs" / stamp
    return p, stamp


def _experiment_description(
    args: argparse.Namespace,
    *,
    layers: list[int],
    run_id: str,
    anchors_dir: Path,
) -> str:
    """One-line human-readable experiment summary for summary.json."""
    compare_txt = (
        "Includes gray-init baseline (--compare)."
        if args.compare
        else "Gray baseline not run."
    )
    init = args.method
    if init == "knn-mixup":
        init_txt = (
            f"KNN-mixup: k={args.knn_k}, similarity={args.knn_sim}, "
            f"score layer={args.match_layer}; post-mix Gaussian blur "
            f"kernel={args.mixup_blur_kernel}, sigma={args.mixup_blur_sigma}."
        )
    elif init == "mixup":
        init_txt = (
            f"Full-anchor pixel mean + blur kernel={args.mixup_blur_kernel}, "
            f"sigma={args.mixup_blur_sigma}."
        )
    else:
        init_txt = (
            f"best: per-layer cosine_mean top-1 anchor at match-layer={args.match_layer}."
        )
    return (
        f"[{run_id}] Qwen3.5-VL ViT feature inversion; see output_dir field. "
        f"model={args.model_name}; layers={layers}; steps={args.steps}; "
        f"restarts={args.restarts}; lr={args.lr}; "
        f"feat/cos/tv/l2={args.feat_weight}/{args.cos_weight}/{args.tv_weight}/{args.l2_weight}; "
        f"method={args.method}; token_match={args.match}; anchors_dir={anchors_dir}; "
        f"{init_txt} {compare_txt}"
    )


from qwen_vision_feature_inversion import (  # noqa: E402
    apply_local_hf_hub_cache,
    hf_hub_local_files_only,
    infer_vision_layer_count,
    parse_layers,
    preprocess_qwen_vl_for_vit,
    resolve_visual_encoder,
    save_image,
    select_tokens,
    spatial_hw_for_qwen_vl_pixels,
    to_logits,
    total_variation,
    vision_hidden_at_layer,
)


def _cosine_mean_pool_features(a: torch.Tensor, b: torch.Tensor, *, match: str) -> float:
    """Cosine similarity of mean-pooled token vectors (same convention as find_best)."""

    def _mp(t: torch.Tensor) -> torch.Tensor:
        t = select_tokens(t, match).float()
        v = t.mean(dim=0) if t.ndim == 2 else t.mean(dim=-2).flatten()
        return v / (v.norm() + 1e-8)

    va, vb = _mp(a), _mp(b)
    return float(torch.dot(va, vb).item())


# ══════════════════════════════════════════════════════════════════════════════
# Anchor Library
# ══════════════════════════════════════════════════════════════════════════════

class AnchorLibrary:
    """
    Manages a small collection of anchor images (one per semantic category).

    Directory layout:
        anchors/
          animal.jpg
          building.jpg
          vehicle.jpg
          ...

    The filename stem (e.g. "animal") is the category label.
    All anchors are bilinearly resized to the *same* spatial resolution as the
    target image (target_size_hw) so their feature tensors are directly
    comparable. We reuse the target's image_grid_thw so positional encodings
    are consistent.

    Features are computed lazily per-layer and cached in memory.
    """

    _EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    def __init__(
        self,
        anchors_dir: Path,
        visual_encoder: Any,
        image_processor: Any,
        target_size_hw: tuple[int, int],
        target_extra_kwargs: dict[str, torch.Tensor],
        device: torch.device,
    ) -> None:
        self.visual_encoder = visual_encoder
        self.image_processor = image_processor
        self.size_hw = target_size_hw
        self.device = device
        # Reuse the target's grid_thw so anchor patches share the same
        # temporal/spatial layout and RoPE positional encoding.
        self._extra = {"image_grid_thw": target_extra_kwargs["image_grid_thw"]}

        self.names: list[str] = []
        self.images_x01: list[torch.Tensor] = []   # each: (1, 3, H, W) in [0,1]
        self._feat_cache: dict[int, list[torch.Tensor]] = {}

        for p in sorted(anchors_dir.iterdir()):
            if p.suffix.lower() not in self._EXTS:
                continue
            try:
                img = Image.open(p).convert("RGB")
                x01 = TF.to_tensor(img).unsqueeze(0)          # (1,3,H_raw,W_raw)
                x01 = F.interpolate(
                    x01, size=target_size_hw,
                    mode="bilinear", align_corners=False,
                ).clamp(0, 1).to(device)
                self.names.append(p.stem)
                self.images_x01.append(x01)
            except Exception as exc:
                print(f"[anchors] skip {p.name}: {exc}")

        if not self.names:
            raise RuntimeError(
                f"No valid anchor images found in {anchors_dir}.\n"
                "Run:  python prepare_anchors.py --anchors-dir anchors/"
            )
        print(f"[anchors] loaded {len(self.names)} categories: {self.names}")

    @torch.no_grad()
    def mixup_uniform_blurred(
        self,
        *,
        blur_kernel_size: int = 5,
        blur_sigma: float = 2.0,
    ) -> torch.Tensor:
        """
        Pixel-space average of all anchors (each already target resolution), then
        Gaussian blur for a softer mixup init. Returns (1, 3, H, W) in [0, 1].
        """
        stacked = torch.cat(self.images_x01, dim=0)
        mixed = stacked.mean(dim=0, keepdim=True).clamp(0, 1)
        k = max(3, int(blur_kernel_size))
        if k % 2 == 0:
            k += 1
        sig = max(0.0, float(blur_sigma))
        if sig == 0:
            return mixed
        return TF.gaussian_blur(mixed, kernel_size=[k, k], sigma=[sig, sig]).clamp(0, 1)

    def _mean_pool_vec(
        self, feat: torch.Tensor, *, match: str, normalize: bool
    ) -> torch.Tensor:
        t = select_tokens(feat, match).float()
        v = t.mean(dim=0) if t.ndim == 2 else t.mean(dim=-2).flatten()
        if normalize:
            return v / (v.norm() + 1e-8)
        return v

    def pairwise_similarities(
        self,
        target_feat: torch.Tensor,
        layer_idx: int,
        *,
        match: str,
        metric: str,
    ) -> list[float]:
        """
        One score per anchor (higher = more similar to target). metric:
          cosine_mean — L2-normalized mean-pool vectors, dot product;
          neg_l2_mean — negative squared L2 between raw mean-pool vectors.
        """
        self._compute_layer(layer_idx)
        if metric == "cosine_mean":
            t = self._mean_pool_vec(target_feat, match=match, normalize=True)
            scores: list[float] = []
            for feat in self._feat_cache[layer_idx]:
                a = self._mean_pool_vec(feat, match=match, normalize=True)
                scores.append(float(torch.dot(t, a).item()))
            return scores
        if metric == "neg_l2_mean":
            t = self._mean_pool_vec(target_feat, match=match, normalize=False)
            scores = []
            for feat in self._feat_cache[layer_idx]:
                a = self._mean_pool_vec(feat, match=match, normalize=False)
                d = t - a
                scores.append(float(-(d * d).sum().item()))
            return scores
        raise ValueError(f"unknown knn metric: {metric}")

    @torch.no_grad()
    def knn_mixup_pixel(
        self,
        target_feat: torch.Tensor,
        sim_layer_idx: int,
        *,
        match: str,
        metric: str,
        k: int,
        blur_kernel_size: int,
        blur_sigma: float,
    ) -> tuple[torch.Tensor, list[tuple[str, float]], str]:
        """
        Encode anchors at sim_layer_idx, rank by similarity to target_feat,
        average the top-k anchor images in pixel space, optional Gaussian blur.

        Returns (mix_x01 (1,3,H,W), [(name, score), ...] sorted best-first, label_str).
        """
        scores = self.pairwise_similarities(
            target_feat, sim_layer_idx, match=match, metric=metric
        )
        n = len(scores)
        kk = min(max(1, int(k)), n)
        order = sorted(range(n), key=lambda i: scores[i], reverse=True)
        top_idx = order[:kk]
        top_pairs = [(self.names[i], scores[i]) for i in top_idx]
        stacked = torch.cat([self.images_x01[i] for i in top_idx], dim=0)
        mixed = stacked.mean(dim=0, keepdim=True).clamp(0, 1)
        kb = max(3, int(blur_kernel_size))
        if kb % 2 == 0:
            kb += 1
        sig = max(0.0, float(blur_sigma))
        if sig > 0:
            mixed = TF.gaussian_blur(mixed, kernel_size=[kb, kb], sigma=[sig, sig]).clamp(0, 1)
        label = "+".join(self.names[i] for i in top_idx)
        return mixed, top_pairs, label

    @torch.no_grad()
    def _compute_layer(self, layer_idx: int) -> None:
        """Encode all anchors at layer_idx and cache the hidden states."""
        if layer_idx in self._feat_cache:
            return
        print(f"[anchors] encoding {len(self.names)} anchors at layer {layer_idx} ...")
        feats: list[torch.Tensor] = []
        for x01 in self.images_x01:
            px = preprocess_qwen_vl_for_vit(
                x01, image_processor=self.image_processor, size_hw=self.size_hw
            )
            feat = vision_hidden_at_layer(
                self.visual_encoder,
                px,
                layer_idx,
                extra_vision_kwargs=self._extra,
                forward_dtype=torch.float32,
            ).float()
            feats.append(feat)
        self._feat_cache[layer_idx] = feats

    def find_best(
        self,
        target_feat: torch.Tensor,
        layer_idx: int,
        match: str = "all",
    ) -> tuple[str, torch.Tensor, float]:
        """
        Return (category_name, anchor_x01, cosine_similarity).

        Similarity is computed on mean-pooled token representations so tensors
        with different sequence lengths (from different image sizes) would still
        be comparable — but here both always share the same size_hw so their
        shapes are identical.
        """
        sims = self.pairwise_similarities(
            target_feat, layer_idx, match=match, metric="cosine_mean",
        )
        best_idx = max(range(len(sims)), key=lambda i: sims[i])
        best_sim = sims[best_idx]

        # Print ranking for transparency
        ranking = sorted(zip(sims, self.names), reverse=True)
        print(
            f"[anchors] layer={layer_idx} similarity ranking: "
            + ", ".join(f"{n}={s:.3f}" for s, n in ranking)
        )
        return self.names[best_idx], self.images_x01[best_idx], float(best_sim)


# ══════════════════════════════════════════════════════════════════════════════
# Initialisation helpers
# ══════════════════════════════════════════════════════════════════════════════

def _gray_init(shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
    """Gray image + small noise (same as optimization.py baseline)."""
    x = torch.full(shape, 0.5, device=device)
    return (x + 0.03 * torch.randn_like(x)).clamp(0, 1)


def _anchor_init(
    anchor_x01: torch.Tensor,
    shape: tuple[int, ...],
    device: torch.device,
    noise_scale: float = 0.02,
) -> torch.Tensor:
    """
    Resize anchor to target spatial size and add small Gaussian noise.
    noise_scale controls the diversity across restarts:
      restart 0  → 0.00  (exact anchor)
      restart 1  → 0.05  (lightly perturbed)
      restart 2+ → 0.10  (more perturbed, wider search)
    """
    h, w = shape[2], shape[3]
    x = F.interpolate(anchor_x01, size=(h, w), mode="bilinear", align_corners=False)
    x = x.to(device)
    if noise_scale > 0:
        x = (x + noise_scale * torch.randn_like(x)).clamp(0, 1)
    return x


# ══════════════════════════════════════════════════════════════════════════════
# Core single-run optimiser
# ══════════════════════════════════════════════════════════════════════════════

def _run_single(
    visual_encoder: Any,
    image_processor: Any,
    target_feat: torch.Tensor,
    layer_idx: int,
    size_hw: tuple[int, int],
    device: torch.device,
    steps: int,
    lr: float,
    feat_weight: float,
    cos_weight: float,
    tv_weight: float,
    l2_weight: float,
    match: str,
    init_x01: torch.Tensor,
    extra_vision_kwargs: dict[str, torch.Tensor] | None,
) -> tuple[torch.Tensor, float]:
    """
    Single gradient-descent run.
    Optimises the logit-parameterised image (logits → sigmoid → x01) with
    AdamW + CosineAnnealingLR. Loss = MSE + cosine + TV + L2-to-gray.
    Returns (best_x01, best_loss).
    """
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")

    logits = to_logits(init_x01.to(device)).requires_grad_(True)
    opt = torch.optim.AdamW([logits], lr=lr, betas=(0.9, 0.99))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.05)

    target = select_tokens(target_feat, match).detach()
    best_loss, best_x = float("inf"), None

    pbar = tqdm(range(steps), desc=f"  layer={layer_idx}", leave=False)
    for _ in pbar:
        opt.zero_grad()
        x01 = torch.sigmoid(logits)
        px = preprocess_qwen_vl_for_vit(
            x01, image_processor=image_processor, size_hw=size_hw
        )
        pred = select_tokens(
            vision_hidden_at_layer(
                visual_encoder,
                px,
                layer_idx,
                extra_vision_kwargs=extra_vision_kwargs,
                forward_dtype=torch.float32,
            ),
            match,
        )
        pf, tf = pred.float(), target.float()
        pb = pf.unsqueeze(0) if pf.ndim == 2 else pf
        tb = tf.unsqueeze(0) if tf.ndim == 2 else tf

        mse  = F.mse_loss(pf, tf)
        cos  = 1.0 - F.cosine_similarity(pb.flatten(1), tb.flatten(1), dim=1).mean()
        tv   = total_variation(x01)
        l2   = (x01 - 0.5).pow(2).mean()
        loss = feat_weight * mse + cos_weight * cos + tv_weight * tv + l2_weight * l2

        if not torch.isfinite(loss):
            print(f"\n  Warning: non-finite loss at layer {layer_idx}, stopping early")
            break

        loss.backward()
        torch.nn.utils.clip_grad_norm_([logits], 1.0)
        opt.step()
        sched.step()

        cur = float(loss.item())
        if cur < best_loss:
            best_loss, best_x = cur, x01.detach().clone()
        pbar.set_postfix(loss=f"{cur:.4f}", mse=f"{mse.item():.4f}")

    final = best_x if best_x is not None else torch.sigmoid(logits).detach()
    return final, best_loss


# ══════════════════════════════════════════════════════════════════════════════
# Multi-restart wrappers
# ══════════════════════════════════════════════════════════════════════════════

# Noise added per restart for anchor init
_ANCHOR_NOISE = [0.00, 0.05, 0.10, 0.15]


def invert_semantic(
    visual_encoder: Any,
    image_processor: Any,
    target_feat: torch.Tensor,
    layer_idx: int,
    size_hw: tuple[int, int],
    device: torch.device,
    steps: int,
    lr: float,
    feat_weight: float,
    cos_weight: float,
    tv_weight: float,
    l2_weight: float,
    match: str,
    restarts: int,
    anchor_x01: torch.Tensor,
    extra_vision_kwargs: dict[str, torch.Tensor] | None,
) -> tuple[torch.Tensor, float]:
    """
    Multi-restart inversion with semantic anchor warm-start.
    Restart 0 starts exactly from the anchor; later restarts add increasing
    noise for diversity.
    """
    best_loss, best_x = float("inf"), None
    for r in range(restarts):
        ns = _ANCHOR_NOISE[min(r, len(_ANCHOR_NOISE) - 1)]
        print(f"  restart {r + 1}/{restarts}  [anchor_init noise={ns:.2f}]")
        init = _anchor_init(anchor_x01, (1, 3, *size_hw), device, ns)
        x_r, loss_r = _run_single(
            visual_encoder, image_processor, target_feat, layer_idx,
            size_hw, device, steps, lr,
            feat_weight, cos_weight, tv_weight, l2_weight,
            match, init, extra_vision_kwargs,
        )
        if loss_r < best_loss:
            best_loss, best_x = loss_r, x_r
    return best_x, best_loss  # type: ignore[return-value]


def invert_gray(
    visual_encoder: Any,
    image_processor: Any,
    target_feat: torch.Tensor,
    layer_idx: int,
    size_hw: tuple[int, int],
    device: torch.device,
    steps: int,
    lr: float,
    feat_weight: float,
    cos_weight: float,
    tv_weight: float,
    l2_weight: float,
    match: str,
    restarts: int,
    extra_vision_kwargs: dict[str, torch.Tensor] | None,
) -> tuple[torch.Tensor, float]:
    """Multi-restart gray-init inversion (baseline)."""
    best_loss, best_x = float("inf"), None
    for r in range(restarts):
        print(f"  restart {r + 1}/{restarts}  [gray_init]")
        init = _gray_init((1, 3, *size_hw), device)
        x_r, loss_r = _run_single(
            visual_encoder, image_processor, target_feat, layer_idx,
            size_hw, device, steps, lr,
            feat_weight, cos_weight, tv_weight, l2_weight,
            match, init, extra_vision_kwargs,
        )
        if loss_r < best_loss:
            best_loss, best_x = loss_r, x_r
    return best_x, best_loss  # type: ignore[return-value]


# ══════════════════════════════════════════════════════════════════════════════
# Method dispatch (add NEW baseline_* / run_* functions; do not edit old ones above)
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class SemanticLayerInit:
    """Per-layer warm-start image + metadata for logging / summary."""

    anchor_x01: torch.Tensor
    anchor_name: str
    anchor_sim: float
    top_pairs: list[tuple[str, float]] = field(default_factory=list)
    row_extra: dict[str, Any] = field(default_factory=dict)


def baseline_mixup_prepare_global(
    anchor_lib: AnchorLibrary,
    out_dir: Path,
    *,
    mixup_blur_kernel: int,
    mixup_blur_sigma: float,
) -> torch.Tensor:
    """Baseline recipe: mean all anchors in pixel space, optional blur; writes diagnostics."""
    stacked_pre = torch.cat(anchor_lib.images_x01, dim=0).mean(dim=0, keepdim=True).clamp(0, 1)
    save_image(stacked_pre, out_dir / "mixup_anchor_preblur.jpg")
    mixup_x01 = anchor_lib.mixup_uniform_blurred(
        blur_kernel_size=mixup_blur_kernel,
        blur_sigma=mixup_blur_sigma,
    )
    save_image(mixup_x01, out_dir / "mixup_anchor_init.jpg")
    print(
        f"[anchors] mixup init: mean of {len(anchor_lib.names)} anchors + blur "
        f"(kernel={mixup_blur_kernel}, sigma={mixup_blur_sigma})"
    )
    return mixup_x01


def baseline_best_semantic_init_for_layer(
    anchor_lib: AnchorLibrary,
    target_feat: torch.Tensor,
    *,
    layer: int,
    match_layer_fixed: int | None,
    match: str,
    out_dir: Path,
) -> SemanticLayerInit:
    """Baseline: per-layer cosine top-1 anchor (uses AnchorLibrary.find_best unchanged)."""
    ml = layer if match_layer_fixed is None else match_layer_fixed
    anchor_name, anchor_x01, anchor_sim = anchor_lib.find_best(
        target_feat, ml, match=match
    )
    print(f"  -> best anchor: '{anchor_name}'  (cosine={anchor_sim:.4f})")
    save_image(anchor_x01, out_dir / f"layer{layer}_chosen_anchor_{anchor_name}.jpg")
    return SemanticLayerInit(
        anchor_x01=anchor_x01,
        anchor_name=anchor_name,
        anchor_sim=float(anchor_sim),
    )


def baseline_mixup_semantic_init_for_layer(
    mixup_x01: torch.Tensor,
    target_feat: torch.Tensor,
    visual_encoder: Any,
    image_processor: Any,
    *,
    layer: int,
    size_hw: tuple[int, int],
    match: str,
    extra_vision_kwargs: dict[str, torch.Tensor],
    n_anchors: int,
) -> SemanticLayerInit:
    """Baseline: same blurred global mixup for every inversion layer."""
    anchor_name = "mixup"
    with torch.no_grad():
        px_m = preprocess_qwen_vl_for_vit(
            mixup_x01,
            image_processor=image_processor,
            size_hw=size_hw,
        )
        feat_mix = vision_hidden_at_layer(
            visual_encoder,
            px_m,
            layer,
            extra_vision_kwargs=extra_vision_kwargs,
            forward_dtype=torch.float32,
        )
    anchor_sim = _cosine_mean_pool_features(
        target_feat, feat_mix, match=match
    )
    print(
        f"  -> anchor init: mixup ({n_anchors} images, blurred); "
        f"target↔mixup feat cosine @ layer {layer} = {anchor_sim:.4f}"
    )
    return SemanticLayerInit(
        anchor_x01=mixup_x01,
        anchor_name=anchor_name,
        anchor_sim=anchor_sim,
    )


def baseline_knn_mixup_semantic_init_for_layer(
    anchor_lib: AnchorLibrary,
    target_feat: torch.Tensor,
    visual_encoder: Any,
    image_processor: Any,
    *,
    layer: int,
    match_layer_fixed: int | None,
    match: str,
    size_hw: tuple[int, int],
    out_dir: Path,
    extra_vision_kwargs: dict[str, torch.Tensor],
    knn_k: int,
    knn_sim: str,
    mixup_blur_kernel: int,
    mixup_blur_sigma: float,
) -> SemanticLayerInit:
    """Baseline: KNN in feature space, then pixel mean of top-k + blur."""
    ml = layer if match_layer_fixed is None else match_layer_fixed
    anchor_x01, top_pairs, label = anchor_lib.knn_mixup_pixel(
        target_feat,
        ml,
        match=match,
        metric=knn_sim,
        k=knn_k,
        blur_kernel_size=mixup_blur_kernel,
        blur_sigma=mixup_blur_sigma,
    )
    anchor_name = f"knn:{label}"
    top_str = ", ".join(f"{n}={s:.3f}" for n, s in top_pairs)
    print(
        f"  -> knn-mixup (k={knn_k}, sim={knn_sim}, score_layer={ml}): {top_str}"
    )
    save_image(
        anchor_x01,
        out_dir / f"layer{layer}_knn_k{knn_k}_{knn_sim.replace('_', '-')}.jpg",
    )
    with torch.no_grad():
        px_m = preprocess_qwen_vl_for_vit(
            anchor_x01,
            image_processor=image_processor,
            size_hw=size_hw,
        )
        feat_mix = vision_hidden_at_layer(
            visual_encoder,
            px_m,
            layer,
            extra_vision_kwargs=extra_vision_kwargs,
            forward_dtype=torch.float32,
        )
    anchor_sim = _cosine_mean_pool_features(
        target_feat, feat_mix, match=match
    )
    print(
        f"    target↔knn-mixup feat cosine @ inversion layer {layer} = {anchor_sim:.4f}"
    )
    row_extra: dict[str, Any] = {
        "knn_k": knn_k,
        "knn_sim": knn_sim,
        "knn_score_layer": ml,
        "knn_top": [{"name": n, "score": round(s, 4)} for n, s in top_pairs],
    }
    return SemanticLayerInit(
        anchor_x01=anchor_x01,
        anchor_name=anchor_name,
        anchor_sim=anchor_sim,
        top_pairs=list(top_pairs),
        row_extra=row_extra,
    )


def run_semantic_layer_init_from_method(
    method: str,
    *,
    layer: int,
    anchor_lib: AnchorLibrary,
    target_feat: torch.Tensor,
    visual_encoder: Any,
    image_processor: Any,
    size_hw: tuple[int, int],
    match_layer_fixed: int | None,
    match: str,
    out_dir: Path,
    extra_vision_kwargs: dict[str, torch.Tensor],
    mixup_x01: torch.Tensor | None,
    knn_k: int,
    knn_sim: str,
    mixup_blur_kernel: int,
    mixup_blur_sigma: float,
) -> SemanticLayerInit:
    """
    Single entry: switch on ``method`` only. New experiments = add a new branch
    (and a new baseline_* function) here without editing older baseline_* bodies.
    """
    if method == "best":
        return baseline_best_semantic_init_for_layer(
            anchor_lib,
            target_feat,
            layer=layer,
            match_layer_fixed=match_layer_fixed,
            match=match,
            out_dir=out_dir,
        )
    if method == "mixup":
        if mixup_x01 is None:
            raise RuntimeError("mixup_x01 required for method=mixup")
        return baseline_mixup_semantic_init_for_layer(
            mixup_x01,
            target_feat,
            visual_encoder,
            image_processor,
            layer=layer,
            size_hw=size_hw,
            match=match,
            extra_vision_kwargs=extra_vision_kwargs,
            n_anchors=len(anchor_lib.names),
        )
    if method == "knn-mixup":
        return baseline_knn_mixup_semantic_init_for_layer(
            anchor_lib,
            target_feat,
            visual_encoder,
            image_processor,
            layer=layer,
            match_layer_fixed=match_layer_fixed,
            match=match,
            size_hw=size_hw,
            out_dir=out_dir,
            extra_vision_kwargs=extra_vision_kwargs,
            knn_k=knn_k,
            knn_sim=knn_sim,
            mixup_blur_kernel=mixup_blur_kernel,
            mixup_blur_sigma=mixup_blur_sigma,
        )
    raise ValueError(
        f"Unknown method={method!r}. Registered: best, mixup, knn-mixup. "
        "Add a new baseline_* and a branch in run_semantic_layer_init_from_method."
    )


# ══════════════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════════════

def compute_psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = F.mse_loss(a.float().clamp(0, 1), b.float().clamp(0, 1)).item()
    return 10.0 * math.log10(1.0 / mse) if mse > 1e-12 else float("inf")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Semantic warm-start feature inversion for Qwen VL vision encoder."
    )
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--image", required=True,
                        help="Path to the target image to invert")
    parser.add_argument("--anchors-dir", default="anchors",
                        help="Directory of anchor images (one per category)")
    parser.add_argument(
        "--layers",
        default="1,4,8,16,last",
        help="ViT hidden_states index: 0=stem, 1..N=after each VisionBlock, last=N (HF output_hidden_states).",
    )
    parser.add_argument("--match-layer", default="same",
                        help="Feature layer for scores: 'same' = inversion layer; fixed int for best/knn-mixup.")
    parser.add_argument(
        "--method",
        choices=["best", "mixup", "knn-mixup"],
        default="knn-mixup",
        help="Warm-start recipe (only change this between runs). Implemented in baseline_* + dispatch.",
    )
    parser.add_argument(
        "--knn-k",
        type=int,
        default=3,
        help="Number of nearest anchors to average (knn-mixup only).",
    )
    parser.add_argument(
        "--knn-sim",
        choices=["cosine_mean", "neg_l2_mean"],
        default="cosine_mean",
        help="Similarity for KNN ranking: cosine on mean-pooled tokens, or neg L2 on mean pools.",
    )
    parser.add_argument(
        "--mixup-blur-kernel",
        type=int,
        default=5,
        help="Gaussian blur kernel after averaging (odd; even +1). Used by mixup and knn-mixup.",
    )
    parser.add_argument(
        "--mixup-blur-sigma",
        type=float,
        default=2.0,
        help="Gaussian blur sigma after averaging; 0 = no blur. mixup and knn-mixup.",
    )
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--feat-weight", type=float, default=1.0)
    parser.add_argument("--cos-weight", type=float, default=0.2)
    parser.add_argument("--tv-weight", type=float, default=1e-3)
    parser.add_argument("--l2-weight", type=float, default=1e-6)
    parser.add_argument("--match", choices=["all", "cls", "patch"], default="all")
    parser.add_argument("--restarts", type=int, default=3)
    parser.add_argument("--compare", action="store_true",
                        help="Also run gray-init baseline for comparison")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        default=None,
        metavar="DIR",
        help="Output directory (relative to semantic_init/). Default: results/runs/<YYYYMMDD-HHMMSS>/",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    out_dir, run_id = _resolve_output_dir(args.output_dir, script_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Experiment output directory: {out_dir}")

    apply_local_hf_hub_cache(script_dir)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── load model (same pattern as qwen_vision_feature_inversion.main) ───────
    print(f"Loading model and processor: {args.model_name}")
    _local = hf_hub_local_files_only()
    processor = AutoProcessor.from_pretrained(
        args.model_name, trust_remote_code=True, local_files_only=_local
    )
    load_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        dtype=load_dtype,
        local_files_only=_local,
    ).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    visual_encoder = resolve_visual_encoder(model)
    image_processor = processor.image_processor

    # ── process target image ─────────────────────────────────────────────────
    raw = Image.open(args.image).convert("RGB")
    proc = image_processor(images=raw, return_tensors="pt")
    pixel_values = proc["pixel_values"].to(device)
    extra_vision_kwargs = {
        k: v.to(device) for k, v in proc.items()
        if k != "pixel_values" and torch.is_tensor(v)
    }
    if "image_grid_thw" not in extra_vision_kwargs:
        raise ValueError(
            "Processor output is missing image_grid_thw; cannot infer spatial size or call the ViT."
        )

    size_hw = spatial_hw_for_qwen_vl_pixels(
        image_processor,
        extra_vision_kwargs["image_grid_thw"],
        image_index=0,
    )
    print(f"Target spatial resolution: {size_hw}")

    # ── layer setup ───────────────────────────────────────────────────────────
    num_layers = infer_vision_layer_count(
        visual_encoder,
        pixel_values,
        extra_vision_kwargs=extra_vision_kwargs,
    )
    layers = parse_layers(args.layers, num_layers)
    print(f"Vision layers: {layers} / total={num_layers}")

    # ── anchor library ────────────────────────────────────────────────────────
    anchors_dir = Path(args.anchors_dir)
    if not anchors_dir.is_absolute():
        anchors_dir = script_dir / anchors_dir
    anchor_lib = AnchorLibrary(
        anchors_dir=anchors_dir,
        visual_encoder=visual_encoder,
        image_processor=image_processor,
        target_size_hw=size_hw,
        target_extra_kwargs=extra_vision_kwargs,
        device=device,
    )

    # ── extract target features ───────────────────────────────────────────────
    with torch.no_grad():
        target_feats = {
            layer: vision_hidden_at_layer(
                visual_encoder,
                pixel_values,
                layer,
                extra_vision_kwargs=extra_vision_kwargs,
            )
            for layer in layers
        }

    # ── save reference image ─────────────────────────────────────────────────
    ref_x01 = TF.to_tensor(raw).unsqueeze(0).to(device)
    ref_x01 = F.interpolate(ref_x01, size=size_hw, mode="bilinear", align_corners=False).clamp(0, 1)
    save_image(ref_x01, out_dir / "reference_resized.jpg")

    mixup_x01: torch.Tensor | None = None
    if args.method == "mixup":
        mixup_x01 = baseline_mixup_prepare_global(
            anchor_lib,
            out_dir,
            mixup_blur_kernel=args.mixup_blur_kernel,
            mixup_blur_sigma=args.mixup_blur_sigma,
        )

    # ── layer used for KNN / best anchor scoring (vs inversion layer) ──────────
    if args.match_layer == "same":
        match_layer_fixed: int | None = None
    else:
        match_layer_fixed = int(args.match_layer)
        if match_layer_fixed not in range(num_layers + 1):
            raise ValueError(f"--match-layer {match_layer_fixed} out of range [0, {num_layers}]")

    # ── inversion loop ────────────────────────────────────────────────────────
    rows: list[dict] = []

    for layer in layers:
        print(f"\n{'=' * 60}")
        print(f"[Inversion] layer = {layer}")

        pack = run_semantic_layer_init_from_method(
            args.method,
            layer=layer,
            anchor_lib=anchor_lib,
            target_feat=target_feats[layer],
            visual_encoder=visual_encoder,
            image_processor=image_processor,
            size_hw=size_hw,
            match_layer_fixed=match_layer_fixed,
            match=args.match,
            out_dir=out_dir,
            extra_vision_kwargs=extra_vision_kwargs,
            mixup_x01=mixup_x01,
            knn_k=args.knn_k,
            knn_sim=args.knn_sim,
            mixup_blur_kernel=args.mixup_blur_kernel,
            mixup_blur_sigma=args.mixup_blur_sigma,
        )
        anchor_x01 = pack.anchor_x01
        anchor_name = pack.anchor_name
        anchor_sim = pack.anchor_sim

        # Semantic warm-start inversion
        print(f"\n  [Semantic init]")
        recon_sem, loss_sem = invert_semantic(
            visual_encoder, image_processor, target_feats[layer], layer,
            size_hw, device, args.steps, args.lr,
            args.feat_weight, args.cos_weight, args.tv_weight, args.l2_weight,
            args.match, args.restarts, anchor_x01, extra_vision_kwargs,
        )
        sem_path = out_dir / f"layer{layer}_semantic.jpg"
        save_image(recon_sem, sem_path)
        psnr_sem = compute_psnr(recon_sem, ref_x01)
        print(f"  Semantic  loss={loss_sem:.6f}  PSNR={psnr_sem:.2f} dB")

        row: dict = {
            "layer": layer,
            "anchor": anchor_name,
            "anchor_cosine": round(anchor_sim, 4),
            "semantic_loss": round(loss_sem, 6),
            "semantic_psnr": round(psnr_sem, 3),
            "semantic_output": str(sem_path),
        }
        if pack.row_extra:
            row.update(pack.row_extra)

        if args.compare:
            print(f"\n  [Gray init  (baseline)]")
            recon_gray, loss_gray = invert_gray(
                visual_encoder, image_processor, target_feats[layer], layer,
                size_hw, device, args.steps, args.lr,
                args.feat_weight, args.cos_weight, args.tv_weight, args.l2_weight,
                args.match, args.restarts, extra_vision_kwargs,
            )
            gray_path = out_dir / f"layer{layer}_gray.jpg"
            save_image(recon_gray, gray_path)
            psnr_gray = compute_psnr(recon_gray, ref_x01)
            print(f"  Gray      loss={loss_gray:.6f}  PSNR={psnr_gray:.2f} dB")
            print(
                f"  PSNR gain: {psnr_sem - psnr_gray:+.2f} dB  "
                f"loss delta: {loss_sem - loss_gray:+.6f}"
            )
            row.update({
                "gray_loss": round(loss_gray, 6),
                "gray_psnr": round(psnr_gray, 3),
                "gray_output": str(gray_path),
                "psnr_gain_db": round(psnr_sem - psnr_gray, 3),
                "loss_delta": round(loss_sem - loss_gray, 6),
            })

        rows.append(row)

    # ── summary ───────────────────────────────────────────────────────────────
    summary = {
        "run_id": run_id,
        "output_dir": str(out_dir.resolve()),
        "experiment_description": _experiment_description(
            args, layers=layers, run_id=run_id, anchors_dir=anchors_dir
        ),
        "model_name": args.model_name,
        "image": args.image,
        "layers": layers,
        "restarts": args.restarts,
        "steps": args.steps,
        "compare": args.compare,
        "method": args.method,
        "match": args.match,
        "match_layer": args.match_layer,
        "results": rows,
    }
    if args.method == "mixup":
        summary["mixup_blur_kernel"] = args.mixup_blur_kernel
        summary["mixup_blur_sigma"] = args.mixup_blur_sigma
    if args.method == "knn-mixup":
        summary["knn_k"] = args.knn_k
        summary["knn_sim"] = args.knn_sim
        summary["mixup_blur_kernel"] = args.mixup_blur_kernel
        summary["mixup_blur_sigma"] = args.mixup_blur_sigma
    summary_path = out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=True)
    print(f"\nSummary saved: {summary_path}")

    if args.compare:
        print("\n── PSNR comparison table ─────────────────────────────")
        print(f"{'layer':>6}  {'anchor':>12}  {'sim':>6}  {'PSNR_sem':>9}  {'PSNR_gray':>9}  {'gain':>6}")
        for r in rows:
            gain_str = f"{r.get('psnr_gain_db', 0):+.2f}" if "psnr_gain_db" in r else "  n/a"
            print(
                f"{r['layer']:>6}  {r['anchor']:>12}  {r['anchor_cosine']:>6.3f}  "
                f"{r['semantic_psnr']:>9.2f}  {r.get('gray_psnr', float('nan')):>9.2f}  "
                f"{gain_str:>6}"
            )


if __name__ == "__main__":
    main()
