"""
Feature inversion from Qwen VL vision encoder hidden states.

Workflow:
1) load Qwen VL model + processor from transformers
2) extract target visual hidden states at selected vision layers
3) optimize an image in [0,1] so its visual features match target features
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor


def parse_layers(layers_text: str, num_layers: int) -> list[int]:
    tokens = [t.strip().lower() for t in layers_text.split(",") if t.strip()]
    if not tokens:
        raise ValueError("--layers cannot be empty")

    layers: list[int] = []
    for token in tokens:
        if token in {"last", "final"}:
            layers.append(num_layers)
            continue
        if not token.isdigit():
            raise ValueError(f"Invalid layer token: {token}")
        layer_idx = int(token)
        if layer_idx < 1:
            raise ValueError(f"Layer must be >=1, got {layer_idx}")
        if layer_idx > num_layers:
            print(f"Warning: skipping layer {layer_idx}; model has {num_layers} layers")
            continue
        layers.append(layer_idx)
    layers = sorted(set(layers))
    if not layers:
        raise ValueError("No valid layers parsed from --layers")
    return layers


def select_tokens(feat: torch.Tensor, match: str) -> torch.Tensor:
    if match == "cls":
        return feat[:, :1, :]
    if match == "patch":
        return feat[:, 1:, :]
    return feat


def total_variation(x: torch.Tensor) -> torch.Tensor:
    dh = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    dw = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    return dh + dw


def to_logits(x01: torch.Tensor) -> torch.Tensor:
    eps = 1e-4
    x = x01.clamp(eps, 1 - eps)
    return torch.log(x) - torch.log1p(-x)


def resolve_visual_encoder(model: Any) -> Any:
    """
    Find vision encoder module from common Qwen VL layouts.
    """
    candidates = [
        ("visual",),
        ("vision_model",),
        ("vision_tower",),
        ("model", "visual"),
        ("model", "vision_model"),
        ("model", "vision_tower"),
    ]
    for path in candidates:
        cur = model
        ok = True
        for key in path:
            if not hasattr(cur, key):
                ok = False
                break
            cur = getattr(cur, key)
        if ok and cur is not None:
            return cur
    raise RuntimeError(
        "Cannot find vision encoder module on this model. "
        "Please check the model architecture and adjust resolve_visual_encoder()."
    )


def preprocess_for_vision(
    x01: torch.Tensor,
    image_processor: Any,
    size_hw: tuple[int, int],
) -> torch.Tensor:
    """
    Minimal differentiable preprocessing aligned with processor mean/std.
    x01 is in [0,1], shape (B,3,H,W).
    """
    h, w = size_hw
    x = F.interpolate(x01, size=(h, w), mode="bilinear", align_corners=False)

    mean = torch.tensor(image_processor.image_mean, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(image_processor.image_std, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    x = (x - mean) / std
    return x


def vision_hidden_at_layer(
    visual_encoder: Any,
    pixel_values: torch.Tensor,
    layer_idx_1based: int,
    extra_vision_kwargs: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    kwargs = dict(extra_vision_kwargs or {})
    out = visual_encoder(pixel_values=pixel_values, output_hidden_states=True, **kwargs)
    return out.hidden_states[layer_idx_1based]


def infer_vision_layer_count(
    visual_encoder: Any,
    pixel_values: torch.Tensor,
    extra_vision_kwargs: dict[str, torch.Tensor] | None = None,
) -> int:
    with torch.no_grad():
        kwargs = dict(extra_vision_kwargs or {})
        out = visual_encoder(pixel_values=pixel_values, output_hidden_states=True, **kwargs)
    # hidden_states[0] is embedding output
    return len(out.hidden_states) - 1


def invert_single_layer(
    visual_encoder: Any,
    image_processor: Any,
    target_feat: torch.Tensor,
    layer_idx: int,
    init_x01: torch.Tensor,
    size_hw: tuple[int, int],
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
    device = target_feat.device
    target = select_tokens(target_feat, match).detach()

    best_loss = float("inf")
    best_x = None

    for r in range(restarts):
        if r == 0:
            x0 = init_x01.clone()
        else:
            x0 = (init_x01 + 0.03 * torch.randn_like(init_x01)).clamp(0, 1)
        logits = to_logits(x0).requires_grad_(True)

        optimizer = torch.optim.AdamW([logits], lr=lr, betas=(0.9, 0.99))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=steps,
            eta_min=lr * 0.05,
        )

        pbar = tqdm(range(steps), desc=f"Vision layer {layer_idx} restart {r + 1}/{restarts}", leave=False)
        for _ in pbar:
            optimizer.zero_grad()
            x01 = torch.sigmoid(logits)
            px = preprocess_for_vision(x01, image_processor=image_processor, size_hw=size_hw)
            pred_feat = vision_hidden_at_layer(
                visual_encoder,
                px,
                layer_idx,
                extra_vision_kwargs=extra_vision_kwargs,
            )
            pred = select_tokens(pred_feat, match)

            feat_mse = F.mse_loss(pred, target)
            feat_cos = 1.0 - F.cosine_similarity(pred.flatten(1), target.flatten(1), dim=1).mean()
            tv = total_variation(x01)
            l2 = (x01 - 0.5).pow(2).mean()

            loss = feat_weight * feat_mse + cos_weight * feat_cos + tv_weight * tv + l2_weight * l2
            loss.backward()
            torch.nn.utils.clip_grad_norm_([logits], 1.0)
            optimizer.step()
            scheduler.step()

            cur_loss = float(loss.item())
            if cur_loss < best_loss:
                best_loss = cur_loss
                best_x = x01.detach().clone()

            pbar.set_postfix(loss=f"{cur_loss:.4f}", mse=f"{feat_mse.item():.4f}", cos=f"{feat_cos.item():.4f}")

    if best_x is None:
        raise RuntimeError("Inversion failed.")
    return best_x, best_loss


def save_image(x01: torch.Tensor, path: Path) -> None:
    arr = x01.squeeze(0).detach().cpu().clamp(0, 1)
    img = Image.fromarray((arr.permute(1, 2, 0).numpy() * 255.0).astype("uint8"))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-VL-3B-Instruct")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--layers", default="1,4,8,last", help="Vision layers, e.g. 1,4,8,last")
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--feat-weight", type=float, default=1.0)
    parser.add_argument("--cos-weight", type=float, default=0.2)
    parser.add_argument("--tv-weight", type=float, default=1e-3)
    parser.add_argument("--l2-weight", type=float, default=1e-6)
    parser.add_argument("--match", choices=["all", "cls", "patch"], default="all")
    parser.add_argument("--restarts", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results/qwen_vision_inversion")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading model and processor: {args.model_name}")
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    ).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    visual_encoder = resolve_visual_encoder(model)
    image_processor = processor.image_processor

    raw = Image.open(args.image).convert("RGB")
    proc = image_processor(images=raw, return_tensors="pt")
    pixel_values = proc["pixel_values"].to(device)
    extra_vision_kwargs = {
        k: v.to(device) for k, v in proc.items() if k != "pixel_values" and torch.is_tensor(v)
    }

    # Determine optimization spatial size from processor output
    _, _, h, w = pixel_values.shape
    size_hw = (h, w)

    num_layers = infer_vision_layer_count(
        visual_encoder,
        pixel_values,
        extra_vision_kwargs=extra_vision_kwargs,
    )
    layers = parse_layers(args.layers, num_layers)
    print(f"Vision layers: {layers} / total={num_layers}")

    # Init image from processor-compatible [0,1] input resolution
    init_np = np.array(raw, dtype=np.float32) / 255.0
    init_x01 = torch.from_numpy(init_np).permute(2, 0, 1).unsqueeze(0).to(device)
    init_x01 = F.interpolate(init_x01, size=size_hw, mode="bilinear", align_corners=False).clamp(0, 1)

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

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_image(init_x01, out_dir / "input_resized.jpg")

    rows = []
    for layer in layers:
        print(f"\n[Inversion] vision layer={layer}")
        recon, loss = invert_single_layer(
            visual_encoder=visual_encoder,
            image_processor=image_processor,
            target_feat=target_feats[layer],
            layer_idx=layer,
            init_x01=init_x01,
            size_hw=size_hw,
            steps=args.steps,
            lr=args.lr,
            feat_weight=args.feat_weight,
            cos_weight=args.cos_weight,
            tv_weight=args.tv_weight,
            l2_weight=args.l2_weight,
            match=args.match,
            restarts=args.restarts,
            extra_vision_kwargs=extra_vision_kwargs,
        )
        out_path = out_dir / f"layer{layer}_reconstructed.jpg"
        save_image(recon, out_path)
        rows.append({"layer": layer, "loss": float(loss), "output": str(out_path)})
        print(f"Saved: {out_path} (loss={loss:.6f})")

    summary = {
        "model_name": args.model_name,
        "image": args.image,
        "layers": layers,
        "results": rows,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSaved summary: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
