"""
Feature inversion for Qwen models using Hugging Face Transformers.

Given a text prompt, this script:
1) extracts hidden states at specified transformer layers
2) optimizes a soft token distribution sequence to match those hidden states
3) decodes the recovered tokens for each layer
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_layers(layers_text: str, num_layers: int) -> list[int]:
    """
    Parse layer config.
    Example: "1,4,8,last" -> [1, 4, 8, num_layers]
    Uses 1-based indexing for transformer blocks.
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
        if layer_idx < 1:
            raise ValueError(f"Layer index must be >= 1, got {layer_idx}")
        if layer_idx > num_layers:
            print(f"Warning: skipping layer {layer_idx}, model has {num_layers} layers")
            continue
        layers.append(layer_idx)

    layers = sorted(set(layers))
    if not layers:
        raise ValueError("No valid layers after parsing --layers")
    return layers


def select_positions(x: torch.Tensor, mode: str) -> torch.Tensor:
    """
    Select sequence positions for matching.
    x: (B, T, D)
    """
    if mode == "first":
        return x[:, :1, :]
    if mode == "last":
        return x[:, -1:, :]
    return x


def get_hidden_at_layer(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor | None = None,
    inputs_embeds: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    layer_idx_1based: int = 1,
) -> torch.Tensor:
    """
    Return hidden states at given 1-based transformer layer.
    hidden_states[0] is embedding output, hidden_states[i] is output of block i.
    """
    outputs = model(
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    return outputs.hidden_states[layer_idx_1based]


def invert_single_layer(
    model: AutoModelForCausalLM,
    target_hidden: torch.Tensor,
    seq_len: int,
    layer_idx: int,
    lr: float,
    steps: int,
    cos_weight: float,
    entropy_weight: float,
    match: str,
    restarts: int,
    temperature: float,
) -> tuple[torch.Tensor, float]:
    """
    Optimize token logits so that hidden states match target_hidden.
    Returns best token ids and best loss.
    """
    device = target_hidden.device
    embed_weight = model.get_input_embeddings().weight  # (V, D)
    vocab_size = embed_weight.shape[0]

    attention_mask = torch.ones((1, seq_len), dtype=torch.long, device=device)
    target = select_positions(target_hidden, match).detach()

    best_loss = float("inf")
    best_ids = None

    for r in range(restarts):
        logits = (0.01 * torch.randn(1, seq_len, vocab_size, device=device)).requires_grad_(True)
        optimizer = torch.optim.AdamW([logits], lr=lr, betas=(0.9, 0.99))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=steps,
            eta_min=lr * 0.05,
        )

        pbar = tqdm(range(steps), desc=f"Layer {layer_idx} restart {r + 1}/{restarts}", leave=False)
        for _ in pbar:
            optimizer.zero_grad()

            probs = torch.softmax(logits / max(temperature, 1e-6), dim=-1)
            soft_embeds = probs @ embed_weight  # (1, T, D)

            pred_hidden = get_hidden_at_layer(
                model=model,
                inputs_embeds=soft_embeds,
                attention_mask=attention_mask,
                layer_idx_1based=layer_idx,
            )
            pred = select_positions(pred_hidden, match)

            mse = F.mse_loss(pred, target)
            cos = 1.0 - F.cosine_similarity(pred.flatten(1), target.flatten(1), dim=1).mean()
            entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=-1).mean()

            loss = mse + cos_weight * cos + entropy_weight * entropy
            loss.backward()
            torch.nn.utils.clip_grad_norm_([logits], 1.0)
            optimizer.step()
            scheduler.step()

            cur_loss = float(loss.item())
            if cur_loss < best_loss:
                best_loss = cur_loss
                best_ids = probs.argmax(dim=-1).detach().clone()

            pbar.set_postfix(loss=f"{cur_loss:.4f}", mse=f"{mse.item():.4f}", cos=f"{cos.item():.4f}")

    if best_ids is None:
        raise RuntimeError("Inversion failed: no valid solution found")
    return best_ids, best_loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-0.6B", help="Hugging Face model id")
    parser.add_argument("--text", required=True, help="Input text used to extract target features")
    parser.add_argument("--layers", default="1,4,8,last", help="Layers for inversion, e.g. 1,4,8,last")
    parser.add_argument("--match", choices=["all", "first", "last"], default="all")
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--cos-weight", type=float, default=0.2)
    parser.add_argument("--entropy-weight", type=float, default=1e-3)
    parser.add_argument("--restarts", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results/qwen_inversion")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading tokenizer and model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    ).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    enc = tokenizer(
        args.text,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_length,
        add_special_tokens=True,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    seq_len = input_ids.shape[1]

    num_layers = model.config.num_hidden_layers
    layers = parse_layers(args.layers, num_layers)

    with torch.no_grad():
        target_hiddens = {
            layer: get_hidden_at_layer(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                layer_idx_1based=layer,
            )
            for layer in layers
        }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    input_text_decoded = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    print(f"Input length: {seq_len} tokens")
    print(f"Layers: {layers}")
    print(f"Input text: {input_text_decoded}")

    rows = []
    for layer in layers:
        print(f"\n[Inversion] layer={layer}")
        rec_ids, best_loss = invert_single_layer(
            model=model,
            target_hidden=target_hiddens[layer],
            seq_len=seq_len,
            layer_idx=layer,
            lr=args.lr,
            steps=args.steps,
            cos_weight=args.cos_weight,
            entropy_weight=args.entropy_weight,
            match=args.match,
            restarts=args.restarts,
            temperature=args.temperature,
        )

        rec_text = tokenizer.decode(rec_ids[0], skip_special_tokens=True)
        row = {
            "layer": layer,
            "loss": best_loss,
            "recovered_text": rec_text,
        }
        rows.append(row)
        print(f"Recovered (layer {layer}): {rec_text}")

        with (out_dir / f"layer{layer}_recovered.txt").open("w", encoding="utf-8") as f:
            f.write(rec_text)

    summary = {
        "model_name": args.model_name,
        "input_text": input_text_decoded,
        "layers": layers,
        "results": rows,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSaved results to: {out_dir}")


if __name__ == "__main__":
    main()
