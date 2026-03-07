"""
Learning-based Feature Inversion: Train decoders to reconstruct images from ViT features.
Uses CIFAR-100 to train decoders for layers 1, 2, 3, 4, 8, 12, then reconstructs test.jpg.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from optimization import (
    get_transform as get_image_transform,
    load_backbone,
    normalize_for_model,
    vit_feature_at_block,
)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ViT-Base: 197 tokens (1 cls + 196 patch), dim 768
NUM_TOKENS = 197
FEAT_DIM = 768
IMG_SIZE = 224


class FeatureDecoder(nn.Module):
    """
    Decode ViT features (B, 197, 768) -> (B, 3, 224, 224).
    Uses patch tokens (14x14) + cls token via small MLP, then deconv upsampling.
    """

    def __init__(self, feat_dim=FEAT_DIM, num_tokens=NUM_TOKENS):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_tokens = num_tokens
        self.patch_tokens = num_tokens - 1  # 196 = 14*14
        self.spatial = 14  # 224/16

        # Lightweight: project each token separately, then reshape patch tokens
        # cls: (B,768) -> (B,768) broadcast to spatial; patch: (B,196,768) -> (B,768,14,14)
        self.cls_proj = nn.Linear(feat_dim, feat_dim)
        self.patch_proj = nn.Linear(feat_dim, feat_dim)
        # Fuse: (B, 768*2, 14, 14) with cls broadcast, then compress
        self.fuse = nn.Sequential(
            nn.Conv2d(feat_dim * 2, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),  # 14 -> 28
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 28 -> 56
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 56 -> 112
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),    # 112 -> 224
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, feat):
        # feat: (B, 197, 768), [0]=cls, [1:]=patch
        B = feat.shape[0]
        cls = feat[:, 0, :]  # (B, 768)
        patch = feat[:, 1:, :]  # (B, 196, 768)
        cls = self.cls_proj(cls)  # (B, 768)
        patch = self.patch_proj(patch)  # (B, 196, 768)
        patch = patch.permute(0, 2, 1).reshape(B, self.feat_dim, self.spatial, self.spatial)
        cls_spatial = cls.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.spatial, self.spatial)
        fused = torch.cat([patch, cls_spatial], dim=1)  # (B, 768*2, 14, 14)
        x = self.fuse(fused)
        x = self.deconv(x)
        return x


def get_cifar_transforms():
    """CIFAR-100 transforms, same as main.py (224x224 for ViT)."""
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(IMG_SIZE, padding=28),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    test_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return train_tf, test_tf


def denormalize(tensor):
    """Normalized tensor -> [0,1]."""
    mean = torch.tensor(IMAGENET_MEAN, device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=tensor.device).view(1, 3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)


def train_decoder(
    backbone,
    decoder,
    layer_idx,
    train_loader,
    device,
    epochs=30,
    lr=1e-3,
    save_path=None,
):
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    decoder = decoder.to(device).train()
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    pbar = tqdm(range(epochs), desc=f"Layer {layer_idx} decoder")
    for epoch in pbar:
        total_loss = 0
        n_batches = 0
        for images, _ in train_loader:
            images = images.to(device)
            with torch.no_grad():
                feats = vit_feature_at_block(backbone, images, layer_idx)
            recon = decoder(feats)
            target_01 = denormalize(images)
            loss = F.mse_loss(recon, target_01) + 0.1 * F.l1_loss(recon, target_01)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        pbar.set_postfix(loss=f"{avg_loss:.4f}")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(decoder.state_dict(), save_path)
        print(f"  Saved decoder to {save_path}")
    return decoder


def reconstruct_with_decoder(backbone, decoder, layer_idx, image_path, device, output_path):
    """Reconstruct image from features using trained decoder."""
    backbone.eval()
    decoder.eval()
    for p in backbone.parameters():
        p.requires_grad = False
    for p in decoder.parameters():
        p.requires_grad = False

    from PIL import Image
    to_pil = transforms.ToPILImage()

    raw = Image.open(image_path).convert("RGB")
    tf = get_image_transform()
    x01 = tf(raw).unsqueeze(0).to(device)
    x_norm = normalize_for_model(x01)

    with torch.no_grad():
        feat = vit_feature_at_block(backbone, x_norm, layer_idx)
        recon = decoder(feat)

    recon = recon.clamp(0, 1).squeeze(0).cpu()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    to_pil(recon).save(output_path)
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "infer", "both"], default="both",
                       help="train: train decoders only; infer: reconstruct only; both: train then infer")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--ckpt-dir", default="./decoders", help="Directory to save/load decoder checkpoints")
    parser.add_argument("--image", default="results/test.jpg", help="Image to reconstruct")
    parser.add_argument("--output-dir", default="results/learning", help="Output directory for reconstructions")
    parser.add_argument("--layers", default="1,2,3,4,8,12", help="Layer indices, comma-separated")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    layers = [int(x.strip()) for x in args.layers.split(",") if x.strip()]
    ckpt_dir = Path(args.ckpt_dir)
    output_dir = Path(args.output_dir)
    image_path = Path(args.image)

    # Load backbone (pretrained ViT, same as optimization)
    print("Loading ViT backbone...")
    backbone = load_backbone(pretrained=True).to(device)
    backbone.eval()
    num_blocks = len(backbone.blocks)
    layers = [l for l in layers if 1 <= l <= num_blocks]
    print(f"Layers: {layers}")

    if args.mode in ("train", "both"):
        train_tf, _ = get_cifar_transforms()
        train_ds = datasets.CIFAR100(
            root=args.data_root,
            train=True,
            download=True,
            transform=train_tf,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=torch.cuda.is_available(),
        )

        for layer_idx in layers:
            print(f"\n===== Training decoder for layer {layer_idx} =====")
            decoder = FeatureDecoder()
            train_decoder(
                backbone=backbone,
                decoder=decoder,
                layer_idx=layer_idx,
                train_loader=train_loader,
                device=device,
                epochs=args.epochs,
                lr=args.lr,
                save_path=ckpt_dir / f"decoder_layer{layer_idx}.pt",
            )

    if args.mode in ("infer", "both"):
        if not image_path.exists():
            print(f"\nWarning: {image_path} not found. Please place test image there.")
            print("Skipping inference.")
            return

        print(f"\n===== Reconstructing {image_path} =====")
        output_dir.mkdir(parents=True, exist_ok=True)

        for layer_idx in layers:
            ckpt_path = ckpt_dir / f"decoder_layer{layer_idx}.pt"
            if not ckpt_path.exists():
                print(f"  Skip layer {layer_idx}: checkpoint not found at {ckpt_path}")
                continue
            decoder = FeatureDecoder()
            decoder.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
            decoder = decoder.to(device)
            out_path = output_dir / f"reconstructed_layer{layer_idx}.jpg"
            reconstruct_with_decoder(backbone, decoder, layer_idx, image_path, device, out_path)
            print(f"  Saved: {out_path}")

        # Save input for comparison
        from PIL import Image
        raw = Image.open(image_path).convert("RGB")
        tf = get_image_transform()
        x01 = tf(raw)
        transforms.ToPILImage()(x01).save(output_dir / "input_224.jpg")
        print(f"  Saved input: {output_dir / 'input_224.jpg'}")
        print(f"\nAll reconstructions saved to: {output_dir}")


if __name__ == "__main__":
    main()
