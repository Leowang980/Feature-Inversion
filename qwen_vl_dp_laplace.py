"""
Laplace noise at chosen Qwen VL ViT hidden indices (same spirit as DP.py):
  1) Invert toward noisy layer features (reuses qwen_vision_feature_inversion).
  2) Continue vision from that layer with noise injected, merge, then VLM generate
     with a Chinese caption prompt.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import qwen_vision_feature_inversion as qv


def laplace_noise_like(x: torch.Tensor, scale: float) -> torch.Tensor:
    if scale <= 0:
        return torch.zeros_like(x)
    dist = torch.distributions.Laplace(
        loc=torch.tensor(0.0, device=x.device, dtype=x.dtype),
        scale=torch.tensor(scale, device=x.device, dtype=x.dtype),
    )
    return dist.sample(x.shape)


def forward_visual_with_laplace(
    visual: Any,
    pixel_values: torch.Tensor,
    grid_thw: torch.Tensor,
    inject_layer_idx: int,
    laplace_scale: float,
    noise_tensor: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Match HF hidden_states indexing: inject_layer_idx = L means the tensor equals
    full forward's hidden_states[L] (L=0: stem; L=k: after ViT block k-1).
    Add noise on that tensor, then run remaining blocks + merger.

    If ``noise_tensor`` is given, it is added at the injection point (same realization
    as e.g. ``clean_hidden + noise_tensor`` for inversion). Otherwise, if
    ``laplace_scale > 0``, samples fresh Laplace noise internally.

    Returns merger output (num_merged_tokens, llm_dim), same layout as
    torch.cat(model.model.get_image_features(...).pooler_output, dim=0) for one image.
    """
    n_blocks = len(visual.blocks)
    if inject_layer_idx < 0 or inject_layer_idx > n_blocks:
        raise ValueError(f"inject_layer_idx must be in [0, {n_blocks}], got {inject_layer_idx}")

    kwargs: dict = {}
    dev_type = pixel_values.device.type
    if dev_type not in ("cuda", "cpu", "xpu"):
        dev_type = "cpu"

    with torch.amp.autocast(device_type=dev_type, enabled=False):
        hidden_states = visual.patch_embed(pixel_values.float())
        pos_embeds = visual.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = visual.rot_pos_emb(grid_thw).reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        for i in range(inject_layer_idx):
            hidden_states = visual.blocks[i](
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        if noise_tensor is not None:
            hidden_states = hidden_states + noise_tensor.to(
                device=hidden_states.device, dtype=hidden_states.dtype
            )
        elif laplace_scale > 0:
            hidden_states = hidden_states + laplace_noise_like(hidden_states, laplace_scale).to(
                hidden_states.dtype
            )

        for i in range(inject_layer_idx, n_blocks):
            hidden_states = visual.blocks[i](
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        merged = visual.merger(hidden_states)

    out_dtype = next(visual.merger.parameters()).dtype
    return merged.to(out_dtype)


def apply_chat_text(processor: Any, user_prompt_zh: str, image: Image.Image) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_prompt_zh},
            ],
        }
    ]
    try:
        return processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def processor_batch(
    processor: Any,
    chat_text: str,
    image: Image.Image,
    device: torch.device,
) -> dict[str, Any]:
    batch = processor(text=[chat_text], images=[image], return_tensors="pt", padding=True)
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def build_inputs_embeds_with_image_features(
    model: Any,
    input_ids: torch.Tensor,
    image_features_flat: torch.Tensor,
) -> torch.Tensor:
    """Scatter precomputed visual tokens into image placeholder positions."""
    inputs_embeds = model.get_input_embeddings()(input_ids)
    image_mask, _ = model.model.get_placeholder_mask(
        input_ids,
        inputs_embeds=inputs_embeds,
        image_features=image_features_flat,
    )
    return inputs_embeds.masked_scatter(
        image_mask,
        image_features_flat.to(inputs_embeds.dtype).reshape(-1),
    )


@torch.inference_mode()
def generate_clean_baseline(
    model: Any,
    processor: Any,
    image: Image.Image,
    user_prompt_zh: str,
    device: torch.device,
    max_new_tokens: int,
) -> str:
    text = apply_chat_text(processor, user_prompt_zh, image)
    inputs = processor_batch(processor, text, image, device)
    model.model.rope_deltas = None
    out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    in_len = inputs["input_ids"].shape[1]
    return processor.batch_decode(out_ids[:, in_len:], skip_special_tokens=True)[0].strip()


@torch.inference_mode()
def generate_with_custom_image_embeds(
    model: Any,
    processor: Any,
    image: Image.Image,
    user_prompt_zh: str,
    device: torch.device,
    image_features_flat: torch.Tensor,
    max_new_tokens: int,
) -> str:
    text = apply_chat_text(processor, user_prompt_zh, image)
    batch = processor_batch(processor, text, image, device)
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    image_grid_thw = batch.get("image_grid_thw")
    mm_token_type_ids = batch.get("mm_token_type_ids")

    inputs_embeds = build_inputs_embeds_with_image_features(model, input_ids, image_features_flat)

    model.model.rope_deltas = None
    # Pass both input_ids and inputs_embeds: HF decoder-only multimodal path keeps full input_ids in
    # model_kwargs so Qwen3-VL can run get_rope_index (M-RoPE); inputs_embeds is used on the first step.
    gen_kw: dict[str, Any] = {
        "input_ids": input_ids,
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "pixel_values": None,
    }
    if image_grid_thw is not None:
        gen_kw["image_grid_thw"] = image_grid_thw
    if mm_token_type_ids is not None:
        gen_kw["mm_token_type_ids"] = mm_token_type_ids

    out_ids = model.generate(**gen_kw)
    # Sequence length matches inputs_embeds
    seq_len = inputs_embeds.shape[1]
    return processor.batch_decode(out_ids[:, seq_len:], skip_special_tokens=True)[0].strip()


def run_experiment(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_dtype = torch.float16 if device.type == "cuda" else torch.float32

    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        dtype=load_dtype,
    ).to(device)
    model.eval()

    visual = qv.resolve_visual_encoder(model)
    image_processor = processor.image_processor
    raw = Image.open(args.image).convert("RGB")

    proc = image_processor(images=raw, return_tensors="pt")
    pixel_values = proc["pixel_values"].to(device)
    extra_vision_kwargs = {
        k: v.to(device) for k, v in proc.items() if k != "pixel_values" and torch.is_tensor(v)
    }
    if "image_grid_thw" not in extra_vision_kwargs:
        raise ValueError("Processor output missing image_grid_thw.")
    grid_thw = extra_vision_kwargs["image_grid_thw"]

    size_hw = qv.spatial_hw_for_qwen_vl_pixels(image_processor, grid_thw, image_index=0)
    num_layers = qv.infer_vision_layer_count(visual, pixel_values, extra_vision_kwargs)
    layers = qv.parse_layers(args.layers, num_layers)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    qv.save_image(
        F.interpolate(
            torchvision_to_x01(raw, device),
            size=size_hw,
            mode="bilinear",
            align_corners=False,
        ).clamp(0, 1),
        out_dir / "reference_resized.jpg",
    )

    user_prompt = args.prompt_zh
    print(f"[Baseline] generating (no Laplace in vision) ...")
    text_clean = generate_clean_baseline(
        model, processor, raw, user_prompt, device, args.max_new_tokens
    )
    print(f"[Baseline] {text_clean[:200]}...")

    summary: dict = {
        "model_name": args.model_name,
        "image": str(args.image),
        "laplace_scale": args.laplace_scale,
        "layers": layers,
        "prompt_zh": user_prompt,
        "baseline_generation": text_clean,
        "per_layer": [],
    }

    for layer in layers:
        print(f"\n===== Layer index {layer} (HF hidden_states[{layer}]) =====")
        with torch.no_grad():
            clean_feat = qv.vision_hidden_at_layer(
                visual, pixel_values, layer, extra_vision_kwargs=extra_vision_kwargs
            )
        noise = laplace_noise_like(clean_feat, args.laplace_scale)
        noisy_feat = clean_feat + noise

        print(
            f"[Inversion] target = clean + same Laplace noise (scale={args.laplace_scale}) at layer {layer}"
        )
        recon, inv_loss = qv.invert_single_layer(
            visual_encoder=visual,
            image_processor=image_processor,
            target_feat=noisy_feat,
            layer_idx=layer,
            size_hw=size_hw,
            device=device,
            steps=args.inv_steps,
            lr=args.inv_lr,
            feat_weight=args.inv_feat_weight,
            cos_weight=args.inv_cos,
            tv_weight=args.inv_tv,
            l2_weight=args.inv_l2,
            match=args.inv_match,
            init=args.inv_init,
            restarts=args.inv_restarts,
            extra_vision_kwargs=extra_vision_kwargs,
        )
        inv_path = out_dir / f"layer{layer}_laplace_inversion.jpg"
        qv.save_image(recon, inv_path)
        print(f"  saved inversion: {inv_path} (loss={inv_loss:.6f})")

        print(f"[VLM] same noise tensor at layer {layer}, then generate ...")
        noisy_merged = forward_visual_with_laplace(
            visual,
            pixel_values,
            grid_thw,
            inject_layer_idx=layer,
            laplace_scale=args.laplace_scale,
            noise_tensor=noise,
        )
        text_noisy = generate_with_custom_image_embeds(
            model,
            processor,
            raw,
            user_prompt,
            device,
            noisy_merged.reshape(-1, noisy_merged.shape[-1]),
            args.max_new_tokens,
        )
        print(f"[VLM noisy] {text_noisy[:200]}...")

        summary["per_layer"].append(
            {
                "layer": layer,
                "inv_loss": float(inv_loss),
                "inversion_image": str(inv_path),
                "generation_with_noisy_vision": text_noisy,
            }
        )

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nDone. Wrote {out_dir / 'summary.json'}")


def torchvision_to_x01(img: Image.Image, device: torch.device) -> torch.Tensor:
    from torchvision.transforms import functional as TF

    return TF.to_tensor(img.convert("RGB")).unsqueeze(0).to(device)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="Qwen/Qwen3.5-4B")
    p.add_argument("--image", type=Path, required=True)
    p.add_argument(
        "--layers",
        default="4",
        help="Comma-separated hidden_states indices (same as qwen_vision_feature_inversion).",
    )
    p.add_argument("--laplace-scale", type=float, default=2.0)
    p.add_argument(
        "--prompt-zh",
        default="详细描述这张图片",
        help="Chinese user prompt for VLM generation.",
    )
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--output-dir", type=Path, default=Path("results/qwen_vl_dp_laplace"))

    p.add_argument("--inv-steps", type=int, default=1200)
    p.add_argument("--inv-lr", type=float, default=0.03)
    p.add_argument("--inv-feat-weight", type=float, default=1.0)
    p.add_argument("--inv-cos", type=float, default=0.2)
    p.add_argument("--inv-tv", type=float, default=1e-3)
    p.add_argument("--inv-l2", type=float, default=1e-6)
    p.add_argument("--inv-match", choices=["all", "cls", "patch"], default="all")
    p.add_argument("--inv-init", choices=["noise", "gray", "zeros"], default="noise")
    p.add_argument("--inv-restarts", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    run_experiment(args)


if __name__ == "__main__":
    main()
