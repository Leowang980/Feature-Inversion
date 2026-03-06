"""
Inject Laplace noise (Local-DP) at specified ViT layers and evaluate:
1) CIFAR-100 classification accuracy
2) Feature inversion comparison using the same method as optimization.py
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import timm
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToPILImage
from tqdm import tqdm

from main import get_transforms
from optimization import (
    feature_inversion,
    get_transform as get_image_transform,
    normalize_for_model,
    parse_stage_blocks,
    vit_feature_at_block,
)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def laplace_noise_like(x: torch.Tensor, scale: float) -> torch.Tensor:
    if scale <= 0:
        return torch.zeros_like(x)
    dist = torch.distributions.Laplace(
        loc=torch.tensor(0.0, device=x.device, dtype=x.dtype),
        scale=torch.tensor(scale, device=x.device, dtype=x.dtype),
    )
    return dist.sample(x.shape)


def forward_with_laplace(
    model: torch.nn.Module,
    x: torch.Tensor,
    noise_layer: int | None,
    laplace_scale: float,
) -> torch.Tensor:
    """
    Inject Laplace noise after the specified block, then continue forward to get logits.
    noise_layer=None means no noise injection.
    """
    x = model.patch_embed(x)
    cls_token = model.cls_token.expand(x.shape[0], -1, -1)
    if getattr(model, "dist_token", None) is None:
        x = torch.cat((cls_token, x), dim=1)
    else:
        dist_token = model.dist_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, dist_token, x), dim=1)
    x = model.pos_drop(x + model.pos_embed)

    for i, blk in enumerate(model.blocks, start=1):
        x = blk(x)
        if noise_layer is not None and i == noise_layer and laplace_scale > 0:
            x = x + laplace_noise_like(x, laplace_scale)

    x = model.norm(x)
    if hasattr(model, "forward_head"):
        return model.forward_head(x, pre_logits=False)
    cls = x[:, 0]
    return model.head(cls)


@torch.no_grad()
def evaluate_accuracy(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    noise_layer: int | None,
    laplace_scale: float,
    max_batches: int = -1,
) -> float:
    model.eval()
    total = 0
    correct = 0
    pbar = tqdm(loader, desc=f"Eval layer={noise_layer}")
    for batch_idx, (images, labels) in enumerate(pbar):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = forward_with_laplace(model, images, noise_layer, laplace_scale)
        preds = logits.argmax(dim=1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
        pbar.set_postfix(acc=f"{100.0 * correct / max(total, 1):.2f}%")
    return 100.0 * correct / max(total, 1)


def psnr(x: torch.Tensor, y: torch.Tensor) -> float:
    mse = F.mse_loss(x, y).item()
    return -10.0 * math.log10(max(mse, 1e-12))


def save_table(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_inversion_with_dp(
    model: torch.nn.Module,
    device: torch.device,
    image_path: Path,
    stages: list[int],
    laplace_scale: float,
    output_dir: Path,
    inv_steps: int,
    inv_lr: float,
    inv_feat_weight: float,
    inv_cos: float,
    inv_tv: float,
    inv_l2: float,
    inv_match: str,
    inv_init: str,
    inv_restarts: int,
) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    to_pil = ToPILImage()

    raw_img = Image.open(image_path).convert("RGB")
    x01 = get_image_transform()(raw_img).unsqueeze(0).to(device)
    x_norm = normalize_for_model(x01)

    # Save input image for reference
    to_pil(x01.squeeze(0).cpu()).save(output_dir / "input_224.jpg")

    rows = []
    with torch.no_grad():
        clean_feats = {layer: vit_feature_at_block(model, x_norm, layer) for layer in stages}

    for layer in stages:
        clean_feat = clean_feats[layer]
        noisy_feat = clean_feat + laplace_noise_like(clean_feat, laplace_scale)

        print(f"\n[Inversion] layer={layer} clean feature")
        recon_clean, loss_clean = feature_inversion(
            backbone=model,
            target_feat=clean_feat,
            block_idx=layer,
            device=device,
            num_steps=inv_steps,
            lr=inv_lr,
            feat_weight=inv_feat_weight,
            cos_weight=inv_cos,
            tv_weight=inv_tv,
            l2_weight=inv_l2,
            match=inv_match,
            init=inv_init,
            restarts=inv_restarts,
        )

        print(f"[Inversion] layer={layer} noisy feature (Laplace)")
        recon_noisy, loss_noisy = feature_inversion(
            backbone=model,
            target_feat=noisy_feat,
            block_idx=layer,
            device=device,
            num_steps=inv_steps,
            lr=inv_lr,
            feat_weight=inv_feat_weight,
            cos_weight=inv_cos,
            tv_weight=inv_tv,
            l2_weight=inv_l2,
            match=inv_match,
            init=inv_init,
            restarts=inv_restarts,
        )

        recon_clean = recon_clean.clamp(0, 1)
        recon_noisy = recon_noisy.clamp(0, 1)
        to_pil(recon_clean.squeeze(0).cpu()).save(output_dir / f"layer{layer}_clean.jpg")
        to_pil(recon_noisy.squeeze(0).cpu()).save(output_dir / f"layer{layer}_laplace.jpg")

        row = {
            "layer": layer,
            "laplace_scale": laplace_scale,
            "inv_loss_clean": float(loss_clean),
            "inv_loss_laplace": float(loss_noisy),
            "mse_clean": float(F.mse_loss(recon_clean, x01).item()),
            "mse_laplace": float(F.mse_loss(recon_noisy, x01).item()),
            "psnr_clean": float(psnr(recon_clean, x01)),
            "psnr_laplace": float(psnr(recon_noisy, x01)),
        }
        rows.append(row)
        print(
            f"  layer={layer} | PSNR clean={row['psnr_clean']:.2f}, "
            f"laplace={row['psnr_laplace']:.2f}"
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="vit_cifar100_best.pt", help="Best model checkpoint saved by main.py")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--stages", default="1,2,3,4,8,12", help="Experiment layers, comma-separated")
    parser.add_argument("--laplace-scale", type=float, default=0.02, help="Laplace noise scale")
    parser.add_argument("--max-test-batches", type=int, default=-1, help="Debug: -1 for full test set")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--run-acc", action="store_true", help="Run CIFAR-100 accuracy experiment")
    parser.add_argument("--run-inversion", action="store_true", help="Run inversion comparison experiment")
    parser.add_argument("--image", default="test.jpg", help="Input image for inversion")
    parser.add_argument("--output-dir", default="results/dp_laplace")

    # Inversion params (aligned with optimization.py)
    parser.add_argument("--inv-steps", type=int, default=1200)
    parser.add_argument("--inv-lr", type=float, default=0.03)
    parser.add_argument("--inv-feat-weight", type=float, default=1.0)
    parser.add_argument("--inv-cos", type=float, default=0.2)
    parser.add_argument("--inv-tv", type=float, default=1e-3)
    parser.add_argument("--inv-l2", type=float, default=1e-6)
    parser.add_argument("--inv-match", choices=["all", "cls", "patch"], default="all")
    parser.add_argument("--inv-init", choices=["noise", "gray", "zeros"], default="gray")
    parser.add_argument("--inv-restarts", type=int, default=3)
    args = parser.parse_args()

    if not args.run_acc and not args.run_inversion:
        args.run_acc = True
        args.run_inversion = True

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=100)
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()

    num_blocks = len(model.blocks)
    stages = parse_stage_blocks(args.stages, num_blocks)
    print(f"Experiment layers: {stages} | Laplace scale={args.laplace_scale}")

    summary = {
        "laplace_scale": args.laplace_scale,
        "stages": stages,
        "accuracy": [],
        "inversion": [],
    }

    if args.run_acc:
        print("\n===== 1) CIFAR-100 Classification =====")
        _, test_tf = get_transforms()
        test_ds = datasets.CIFAR100(
            root=args.data_root, train=False, download=True, transform=test_tf
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=torch.cuda.is_available(),
        )

        baseline_acc = evaluate_accuracy(
            model=model,
            loader=test_loader,
            device=device,
            noise_layer=None,
            laplace_scale=0.0,
            max_batches=args.max_test_batches,
        )
        summary["accuracy"].append(
            {"layer": "baseline", "laplace_scale": 0.0, "acc": baseline_acc}
        )
        print(f"baseline acc: {baseline_acc:.2f}%")

        rows_acc = []
        for layer in stages:
            acc = evaluate_accuracy(
                model=model,
                loader=test_loader,
                device=device,
                noise_layer=layer,
                laplace_scale=args.laplace_scale,
                max_batches=args.max_test_batches,
            )
            row = {"layer": layer, "laplace_scale": args.laplace_scale, "acc": acc}
            rows_acc.append(row)
            summary["accuracy"].append(row)
            print(f"layer {layer} + Laplace(scale={args.laplace_scale}) -> acc {acc:.2f}%")
        save_table(rows_acc, output_dir / "accuracy_by_layer.csv")

    if args.run_inversion:
        print("\n===== 2) Inversion + Laplace Experiment =====")
        rows_inv = run_inversion_with_dp(
            model=model,
            device=device,
            image_path=Path(args.image),
            stages=stages,
            laplace_scale=args.laplace_scale,
            output_dir=output_dir,
            inv_steps=args.inv_steps,
            inv_lr=args.inv_lr,
            inv_feat_weight=args.inv_feat_weight,
            inv_cos=args.inv_cos,
            inv_tv=args.inv_tv,
            inv_l2=args.inv_l2,
            inv_match=args.inv_match,
            inv_init=args.inv_init,
            inv_restarts=args.inv_restarts,
        )
        summary["inversion"].extend(rows_inv)
        save_table(rows_inv, output_dir / "inversion_by_layer.csv")

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
