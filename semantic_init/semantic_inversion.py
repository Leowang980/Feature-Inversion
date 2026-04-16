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
   animal.jpg, building.jpg …).
2. For each target layer, encode all anchors at that layer and select the one
   with the highest cosine similarity to the target features (mean-pooled over
   the token dimension).
3. Warm-start the inversion from the chosen anchor image.
4. Optionally also run a gray-init baseline (--compare) to measure the gain.

Usage
-----
    If the repo (or any ancestor directory) contains models/hub/ with a HF hub
    cache for Qwen/Qwen3.5-4B, HF_HUB_CACHE is set automatically so weights load
    from disk.

    # First time: set up anchor images
    python prepare_anchors.py --anchors-dir anchors/

    # Run semantic inversion (+ comparison)
    python semantic_inversion.py \\
        --model-name Qwen/Qwen3.5-4B \\
        --image /path/to/target.jpg \\
        --anchors-dir anchors/ \\
        --layers 1,4,8,16,last \\
        --steps 1500 \\
        --restarts 3 \\
        --compare
"""
from __future__ import annotations

import argparse
import json
import math
import sys
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
from qwen_vision_feature_inversion import (  # noqa: E402
    apply_local_hf_hub_cache,
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
          …

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
    def _compute_layer(self, layer_idx: int) -> None:
        """Encode all anchors at layer_idx and cache the hidden states."""
        if layer_idx in self._feat_cache:
            return
        print(f"[anchors] encoding {len(self.names)} anchors at layer {layer_idx} …")
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
        self._compute_layer(layer_idx)

        def mean_pool(t: torch.Tensor) -> torch.Tensor:
            t = select_tokens(t, match).float()
            # t is (seq_len, dim) or (B, seq_len, dim)
            v = t.mean(dim=0) if t.ndim == 2 else t.mean(dim=-2).flatten()
            return v / (v.norm() + 1e-8)

        target_vec = mean_pool(target_feat)
        best_idx, best_sim = 0, -float("inf")
        sims: list[float] = []
        for i, feat in enumerate(self._feat_cache[layer_idx]):
            sim = float(torch.dot(target_vec, mean_pool(feat)).item())
            sims.append(sim)
            if sim > best_sim:
                best_sim, best_idx = sim, i

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
                        help="Layer used for anchor matching. 'same' = use the inversion layer. "
                             "Or specify a fixed index, e.g. '8'.")
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
    parser.add_argument("--output-dir", default="results/semantic_inversion")
    args = parser.parse_args()

    apply_local_hf_hub_cache(Path(__file__).resolve().parent)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── load model (same pattern as qwen_vision_feature_inversion.main) ───────
    print(f"Loading model and processor: {args.model_name}")
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    load_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        dtype=load_dtype,
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
        anchors_dir = Path(__file__).parent / anchors_dir
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
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_x01 = TF.to_tensor(raw).unsqueeze(0).to(device)
    ref_x01 = F.interpolate(ref_x01, size=size_hw, mode="bilinear", align_corners=False).clamp(0, 1)
    save_image(ref_x01, out_dir / "reference_resized.jpg")

    # ── determine anchor-matching layer ───────────────────────────────────────
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

        # Select anchor for this layer
        ml = layer if match_layer_fixed is None else match_layer_fixed
        anchor_name, anchor_x01, anchor_sim = anchor_lib.find_best(
            target_feats[layer], ml, match=args.match
        )
        print(f"  -> best anchor: '{anchor_name}'  (cosine={anchor_sim:.4f})")

        # Save the chosen anchor for inspection
        save_image(anchor_x01, out_dir / f"layer{layer}_chosen_anchor_{anchor_name}.jpg")

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
        "model_name": args.model_name,
        "image": args.image,
        "layers": layers,
        "restarts": args.restarts,
        "steps": args.steps,
        "compare": args.compare,
        "results": rows,
    }
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
