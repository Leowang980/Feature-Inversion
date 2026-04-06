"""
Feature inversion from Qwen VL vision encoder hidden states.

Workflow:
1) load Qwen VL model + processor from transformers
2) extract target visual hidden states at selected indices in output_hidden_states
   （与 HF Qwen3_VL / Qwen3_5 ViT 的 capture_outputs 约定一致：0=stem，1..N=各 block 后）
3) optimize an image in [0,1] so its visual features match target features
   （优化流程对齐 `optimization.py`：init_image + 多 restart 取最优；默认不从原图 warm-start）
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor


def parse_layers(layers_text: str, num_layers: int) -> list[int]:
    """
    与 HF output_hidden_states 对齐（见 modeling_qwen3_5 + output_capturing）：
      - 0: patch_embed + 位置编码之后、进入第 1 个 VisionBlock 之前
      - 1..num_layers: 第 k 个 VisionBlock 的输出（k 从 1 计）
    num_layers = len(hidden_states) - 1，即 ViT block 数；last == num_layers。
    """
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
        if layer_idx < 0:
            raise ValueError(f"Layer must be >=0, got {layer_idx}")
        if layer_idx > num_layers:
            print(f"Warning: skipping layer {layer_idx}; valid indices are 0..{num_layers}")
            continue
        layers.append(layer_idx)
    layers = sorted(set(layers))
    if not layers:
        raise ValueError("No valid layers parsed from --layers")
    return layers


def select_tokens(feat: torch.Tensor, match: str) -> torch.Tensor:
    """
    Qwen3.5 ViT 输出为 (seq_len, hidden_dim)，无 batch 维；与部分 CLIP 式 (B, 1+N, D) 不同。
    --match cls/patch 仅适用于带 CLS 的编码器；此处对 2D 特征做「首 token / 其余 token」切片以便调试。
    """
    if feat.ndim == 2:
        if match == "cls":
            return feat[:1, :]
        if match == "patch":
            return feat[1:, :]
        return feat
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


def init_image(shape: tuple[int, ...], device: torch.device, init: str = "gray") -> torch.Tensor:
    """与 `optimization.init_image` 一致：在 [0,1] 空间初始化 (1,3,H,W)。"""
    if init == "noise":
        x0 = torch.rand(shape, device=device)
    elif init == "zeros":
        x0 = torch.zeros(shape, device=device)
    else:
        x0 = torch.full(shape, 0.5, device=device)
        x0 = (x0 + 0.03 * torch.randn(shape, device=device)).clamp(0, 1)
    return x0


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


def pack_qwen_vl_pixel_values(
    x: torch.Tensor,
    *,
    patch_size: int,
    temporal_patch_size: int,
    merge_size: int,
) -> torch.Tensor:
    """
    将 (B, 3, H, W) 的归一化图像张量排成与 HF ``Qwen2VLImageProcessor`` 相同的
    ``(sum_i patches_i, C*T*patch^2)`` 布局，供 Qwen VL / Qwen3.5 ViT 的 ``patch_embed`` 使用。
    """
    if x.ndim != 4:
        raise ValueError(f"pack_qwen_vl_pixel_values 需要 (B,3,H,W)，当前 {tuple(x.shape)}")
    b, c, h, w = x.shape
    if c != 3:
        raise ValueError("仅支持 3 通道输入")
    if h % patch_size or w % patch_size:
        raise ValueError(f"H、W 需被 patch_size={patch_size} 整除，当前 {(h, w)}")
    gh, gw = h // patch_size, w // patch_size
    if gh % merge_size or gw % merge_size:
        raise ValueError(
            f"patch 格点 (gh,gw)=({gh},{gw}) 需被 merge_size={merge_size} 整除；"
            "请使用与 processor smart_resize 一致的分辨率。"
        )

    patches = x.unsqueeze(1)
    if patches.shape[1] % temporal_patch_size != 0:
        need = temporal_patch_size - patches.shape[1]
        rep = patches[:, -1:].repeat(1, need, 1, 1, 1)
        patches = torch.cat([patches, rep], dim=1)

    batch_size = b
    grid_t = patches.shape[1] // temporal_patch_size
    patches = patches.reshape(
        batch_size,
        grid_t,
        temporal_patch_size,
        c,
        gh // merge_size,
        merge_size,
        patch_size,
        gw // merge_size,
        merge_size,
        patch_size,
    )
    patches = patches.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
    flat = patches.reshape(
        batch_size,
        grid_t * gh * gw,
        c * temporal_patch_size * patch_size * patch_size,
    )
    return flat.reshape(-1, c * temporal_patch_size * patch_size * patch_size)


def preprocess_qwen_vl_for_vit(
    x01: torch.Tensor,
    image_processor: Any,
    size_hw: tuple[int, int],
) -> torch.Tensor:
    """
    x01 为 [0,1]、形状 (B,3,*,*)：双线性到 processor 空间尺寸，CLIP 式 mean/std，
    再打包为 ViT 所需的 ``(num_patches, dim)``（不可再使用 (B,3,H,W) 直接喂 ViT）。
    """
    h, w = size_hw
    x = F.interpolate(x01, size=(h, w), mode="bilinear", align_corners=False)
    mean = torch.tensor(image_processor.image_mean, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(image_processor.image_std, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    x = (x - mean) / std
    ps = int(getattr(image_processor, "patch_size", 14))
    ts = int(getattr(image_processor, "temporal_patch_size", 2))
    ms = int(getattr(image_processor, "merge_size", 2))
    return pack_qwen_vl_pixel_values(x, patch_size=ps, temporal_patch_size=ts, merge_size=ms)


def spatial_hw_for_qwen_vl_pixels(
    image_processor: Any,
    image_grid_thw: torch.Tensor,
    *,
    image_index: int = 0,
) -> tuple[int, int]:
    """
    Qwen2VL / Qwen3VL（及 Qwen3.5-4B 等）的 image_processor 返回的 pixel_values 为
    ``(num_patches, patch_dim)`` 的 patch 序列，不是 ``(B, 3, H, W)``。
    与 processor 一致的空间尺寸 = 空间方向 patch 格点数 × patch_size
    （对应 HF 内部 smart_resize 后的 ``grid_h, grid_w``）。
    """
    patch_size = int(getattr(image_processor, "patch_size", 14))
    gtw = image_grid_thw
    if gtw.ndim == 1:
        g = gtw
    else:
        g = gtw[image_index]
    grid_h = int(g[1].item())
    grid_w = int(g[2].item())
    return grid_h * patch_size, grid_w * patch_size


def call_qwen_vision_encoder(
    visual_encoder: Any,
    pixel_values: torch.Tensor,
    extra_vision_kwargs: dict[str, torch.Tensor] | None = None,
    *,
    output_hidden_states: bool = True,
) -> Any:
    """
    Qwen3_5VisionModel.forward 签名为 (hidden_states, grid_thw)，不能使用 pixel_values=。
    Processor 提供 image_grid_thw，需映射为 grid_thw；其余 tensor 键不要传入 ViT，以免 **kwargs 传入子模块报错。
    """
    extra = dict(extra_vision_kwargs or {})
    grid = extra.pop("image_grid_thw", None)
    if grid is None:
        grid = extra.pop("grid_thw", None)
    if grid is None:
        raise ValueError(
            "缺少 image_grid_thw（或 grid_thw）。请使用与官方一致的 processor 输出，或显式传入 grid。"
        )
    for drop in (
        "pixel_values",
        "pixel_values_videos",
        "video_grid_thw",
        "input_ids",
        "attention_mask",
    ):
        extra.pop(drop, None)

    dtype = getattr(visual_encoder, "dtype", None)
    if dtype is None:
        dtype = next(visual_encoder.parameters()).dtype
    pixel_values = pixel_values.to(dtype=dtype)

    return visual_encoder(
        pixel_values,
        grid_thw=grid,
        output_hidden_states=output_hidden_states,
        return_dict=True,
        **extra,
    )


def vision_hidden_at_layer(
    visual_encoder: Any,
    pixel_values: torch.Tensor,
    layer_idx: int,
    extra_vision_kwargs: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    out = call_qwen_vision_encoder(
        visual_encoder,
        pixel_values,
        extra_vision_kwargs,
        output_hidden_states=True,
    )
    return out.hidden_states[layer_idx]


def infer_vision_layer_count(
    visual_encoder: Any,
    pixel_values: torch.Tensor,
    extra_vision_kwargs: dict[str, torch.Tensor] | None = None,
) -> int:
    """ViT Transformer 层数 N；hidden_states 共 N+1 项（含 stem 的 [0]）。"""
    with torch.no_grad():
        out = call_qwen_vision_encoder(
            visual_encoder,
            pixel_values,
            extra_vision_kwargs,
            output_hidden_states=True,
        )
    return len(out.hidden_states) - 1


def _single_restart_qwen_vision(
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
    init: str,
    extra_vision_kwargs: dict[str, torch.Tensor] | None,
) -> tuple[torch.Tensor, float]:
    """
    单次 restart，与 `optimization._single_restart` 对齐：AdamW + CosineAnnealingLR、
    MSE+cosine+TV+L2、grad clip、记录本 run 最优 x。
    """
    h, w = size_hw
    x0 = init_image((1, 3, h, w), device=device, init=init)
    logits = to_logits(x0).requires_grad_(True)

    optimizer = torch.optim.AdamW([logits], lr=lr, betas=(0.9, 0.99))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=steps,
        eta_min=lr * 0.05,
    )

    best_loss = float("inf")
    best_x = None
    target = select_tokens(target_feat, match).detach()

    pbar = tqdm(range(steps), desc=f"Vision layer {layer_idx} inversion", leave=False)
    for _ in pbar:
        optimizer.zero_grad()
        x01 = torch.sigmoid(logits)
        px = preprocess_qwen_vl_for_vit(x01, image_processor=image_processor, size_hw=size_hw)
        pred_feat = vision_hidden_at_layer(
            visual_encoder,
            px,
            layer_idx,
            extra_vision_kwargs=extra_vision_kwargs,
        )
        pred = select_tokens(pred_feat, match)

        feat_mse = F.mse_loss(pred, target)
        pred_b = pred.unsqueeze(0) if pred.ndim == 2 else pred
        target_b = target.unsqueeze(0) if target.ndim == 2 else target
        feat_cos = 1.0 - F.cosine_similarity(pred_b.flatten(1), target_b.flatten(1), dim=1).mean()
        tv_loss = total_variation(x01)
        l2_loss = (x01 - 0.5).pow(2).mean()

        loss = (
            feat_weight * feat_mse
            + cos_weight * feat_cos
            + tv_weight * tv_loss
            + l2_weight * l2_loss
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_([logits], 1.0)
        optimizer.step()
        scheduler.step()

        cur_loss = loss.item()
        if cur_loss < best_loss:
            best_loss = cur_loss
            best_x = x01.detach().clone()

        pbar.set_postfix(
            loss=f"{cur_loss:.4f}",
            mse=f"{feat_mse.item():.4f}",
            cos=f"{feat_cos.item():.4f}",
        )

    if best_x is None:
        raise RuntimeError("Inversion failed.")
    return best_x, best_loss


def invert_single_layer(
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
    init: str,
    restarts: int,
    extra_vision_kwargs: dict[str, torch.Tensor] | None,
) -> tuple[torch.Tensor, float]:
    """多 restart，与 `optimization.feature_inversion` 一致：取 loss 最低的一次结果。"""
    best_loss = float("inf")
    best_x = None
    for r in range(restarts):
        print(f"  - restart {r + 1}/{restarts}")
        x_r, loss_r = _single_restart_qwen_vision(
            visual_encoder=visual_encoder,
            image_processor=image_processor,
            target_feat=target_feat,
            layer_idx=layer_idx,
            size_hw=size_hw,
            device=device,
            steps=steps,
            lr=lr,
            feat_weight=feat_weight,
            cos_weight=cos_weight,
            tv_weight=tv_weight,
            l2_weight=l2_weight,
            match=match,
            init=init,
            extra_vision_kwargs=extra_vision_kwargs,
        )
        if loss_r < best_loss:
            best_loss = loss_r
            best_x = x_r
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
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--image", default="/root/autodl-tmp/WechatIMG26.jpg", help="输入图像路径（仅用于抽取 target 特征与保存参考缩略图）")
    parser.add_argument(
        "--layers",
        default="1,4,8,last",
        help="ViT hidden_states 索引: 0=stem, 1..N=各 VisionBlock 后, last=N（与 HF output_hidden_states 一致）",
    )
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--feat-weight", type=float, default=1.0)
    parser.add_argument("--cos-weight", type=float, default=0.2)
    parser.add_argument("--tv-weight", type=float, default=1e-3)
    parser.add_argument("--l2-weight", type=float, default=1e-6)
    parser.add_argument("--match", choices=["all", "cls", "patch"], default="all")
    parser.add_argument(
        "--init",
        choices=["noise", "gray", "zeros"],
        default="noise",
        help="与 optimization.py 相同：每次 restart 的 (1,3,H,W) 初值；默认 noise（均匀随机 [0,1]）",
    )
    parser.add_argument("--restarts", type=int, default=3, help="与 optimization.feature_inversion 默认一致")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results/qwen_vision_inversion")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

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
    # print(visual_encoder)
    # print(image_processor)
    # exit()
    raw = Image.open(args.image).convert("RGB")
    proc = image_processor(images=raw, return_tensors="pt")
    pixel_values = proc["pixel_values"].to(device)
    extra_vision_kwargs = {
        k: v.to(device) for k, v in proc.items() if k != "pixel_values" and torch.is_tensor(v)
    }

    if "image_grid_thw" not in extra_vision_kwargs:
        raise ValueError("processor 输出中缺少 image_grid_thw，无法推断空间尺寸或调用 ViT。")
    # pixel_values: (num_patches, dim)，不能解包成 (B,C,H,W)
    size_hw = spatial_hw_for_qwen_vl_pixels(
        image_processor,
        extra_vision_kwargs["image_grid_thw"],
        image_index=0,
    )

    num_layers = infer_vision_layer_count(
        visual_encoder,
        pixel_values,
        extra_vision_kwargs=extra_vision_kwargs,
    )
    layers = parse_layers(args.layers, num_layers)
    print(f"Vision layers: {layers} / total={num_layers}")

    # 仅保存与 ViT 输入同分辨率的参考图，不参与优化初值
    ref_x01 = TF.to_tensor(raw).unsqueeze(0).to(device)
    ref_x01 = F.interpolate(ref_x01, size=size_hw, mode="bilinear", align_corners=False).clamp(0, 1)

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
    save_image(ref_x01, out_dir / "reference_resized.jpg")

    rows = []
    for layer in layers:
        print(f"\n[Inversion] vision layer={layer}")
        recon, loss = invert_single_layer(
            visual_encoder=visual_encoder,
            image_processor=image_processor,
            target_feat=target_feats[layer],
            layer_idx=layer,
            size_hw=size_hw,
            device=device,
            steps=args.steps,
            lr=args.lr,
            feat_weight=args.feat_weight,
            cos_weight=args.cos_weight,
            tv_weight=args.tv_weight,
            l2_weight=args.l2_weight,
            match=args.match,
            init=args.init,
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
        "init": args.init,
        "restarts": args.restarts,
        "results": rows,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSaved summary: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
