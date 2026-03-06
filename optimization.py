"""
Feature Inversion: Reconstruct images from ViT intermediate layer features.
Supports inversion at outputs of blocks 4, 8, or the last block.
"""
import torch
import torch.nn.functional as F
import timm
from torchvision import transforms
from PIL import Image
from tqdm import tqdm


# Same ImageNet normalization as main.py
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_backbone(pretrained=True):
    """Load the same ViT backbone as main.py, without classifier."""
    model = timm.create_model(
        "vit_base_patch16_224",
        pretrained=pretrained,
        num_classes=0,  # Remove classifier head
    )
    return model


def get_transform():
    """Load image and resize to ViT input size (no normalization)."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])


def total_variation(x):
    """TV regularization to encourage smooth images."""
    diff_h = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    diff_w = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    return diff_h + diff_w


def normalize_for_model(x):
    """[0,1] image -> ImageNet normalization."""
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def to_logits(x01):
    """Convert [0,1] image to optimizable logits (sigmoid constrains range)."""
    eps = 1e-4
    x = x01.clamp(eps, 1 - eps)
    return torch.log(x) - torch.log1p(-x)


def init_image(shape, device, init="gray"):
    """Initialize image in [0,1] space."""
    if init == "noise":
        x0 = torch.rand(shape, device=device)
    elif init == "zeros":
        x0 = torch.zeros(shape, device=device)
    else:
        # Gray init is more stable, reduces strong noise artifacts
        x0 = torch.full(shape, 0.5, device=device)
        x0 = (x0 + 0.03 * torch.randn(shape, device=device)).clamp(0, 1)
    return x0


def select_tokens(feat, match):
    """Select token subset for matching."""
    if match == "cls":
        return feat[:, :1, :]
    if match == "patch":
        return feat[:, 1:, :]
    return feat


def vit_feature_at_block(backbone, x_norm, block_idx):
    """
    Forward to specified block and return its output (with norm).
    block_idx: 1-based index.
    """
    x = backbone.patch_embed(x_norm)
    cls_token = backbone.cls_token.expand(x.shape[0], -1, -1)
    if getattr(backbone, "dist_token", None) is None:
        x = torch.cat((cls_token, x), dim=1)
    else:
        dist_token = backbone.dist_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, dist_token, x), dim=1)
    x = backbone.pos_drop(x + backbone.pos_embed)

    for i, blk in enumerate(backbone.blocks, start=1):
        x = blk(x)
        if i == block_idx:
            return backbone.norm(x)
    raise ValueError(f"block_idx={block_idx} out of range, model has {len(backbone.blocks)} blocks")


def _single_restart(
    backbone,
    target_feat,
    block_idx,
    device,
    num_steps,
    lr,
    feat_weight,
    cos_weight,
    tv_weight,
    l2_weight,
    match,
    init,
):
    x0 = init_image((1, 3, 224, 224), device=device, init=init)
    logits = to_logits(x0).requires_grad_(True)

    optimizer = torch.optim.AdamW([logits], lr=lr, betas=(0.9, 0.99))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_steps,
        eta_min=lr * 0.05,
    )

    best_loss = float("inf")
    best_x = None

    pbar = tqdm(range(num_steps), desc=f"Block {block_idx} inversion", leave=False)
    for _ in pbar:
        optimizer.zero_grad()
        x01 = torch.sigmoid(logits)
        x_norm = normalize_for_model(x01)

        pred_feat = vit_feature_at_block(backbone, x_norm, block_idx)
        pred = select_tokens(pred_feat, match)
        target = select_tokens(target_feat, match)

        feat_mse = F.mse_loss(pred, target)
        feat_cos = 1.0 - F.cosine_similarity(
            pred.flatten(1), target.flatten(1), dim=1
        ).mean()
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

    return best_x, best_loss


def feature_inversion(
    backbone,
    target_feat,
    block_idx,
    device,
    num_steps=1200,
    lr=0.03,
    feat_weight=1.0,
    cos_weight=0.2,
    tv_weight=1e-3,
    l2_weight=1e-6,
    match="all",
    init="gray",
    restarts=3,
):
    """Reconstruct image from target_feat at specified block via multi-restart optimization."""
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    best_loss = float("inf")
    best_x = None
    for r in range(restarts):
        print(f"  - restart {r + 1}/{restarts}")
        x_r, loss_r = _single_restart(
            backbone=backbone,
            target_feat=target_feat,
            block_idx=block_idx,
            device=device,
            num_steps=num_steps,
            lr=lr,
            feat_weight=feat_weight,
            cos_weight=cos_weight,
            tv_weight=tv_weight,
            l2_weight=l2_weight,
            match=match,
            init=init if init != "zeros" else "zeros",
        )
        if loss_r < best_loss:
            best_loss = loss_r
            best_x = x_r

    return best_x, best_loss


def denormalize(tensor):
    """Convert normalized tensor back to [0,1]."""
    mean = torch.tensor(IMAGENET_MEAN, device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=tensor.device).view(1, 3, 1, 1)
    return tensor * std + mean


def parse_stage_blocks(stages_text, num_blocks):
    """
    Parse stage configuration.
    E.g. "1,2,3,4,8,last" -> [1,2,3,4,8,num_blocks]
    """
    tokens = [t.strip().lower() for t in stages_text.split(",") if t.strip()]
    if not tokens:
        raise ValueError("--stages cannot be empty")

    stage_blocks = []
    for token in tokens:
        if token in {"last", "final"}:
            stage_blocks.append(num_blocks)
            continue
        if not token.isdigit():
            raise ValueError(f"Invalid stage '{token}', only positive integers or 'last' supported")
        block_idx = int(token)
        if block_idx < 1:
            raise ValueError(f"Stage must be >= 1, got {block_idx}")
        if block_idx > num_blocks:
            print(f"Warning: skipping block {block_idx} (model has only {num_blocks} blocks)")
            continue
        stage_blocks.append(block_idx)

    stage_blocks = sorted(set(stage_blocks))
    if not stage_blocks:
        raise ValueError("No valid stages, check --stages configuration")
    return stage_blocks


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="test.jpg", help="Input image path")
    parser.add_argument("--output", default=None, help="Legacy: save path for last-stage reconstruction")
    parser.add_argument("--output-prefix", default="reconstructed", help="Prefix for multi-stage outputs")
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--feat-weight", type=float, default=1.0)
    parser.add_argument("--cos", type=float, default=0.2, help="Cosine loss weight")
    parser.add_argument("--tv", type=float, default=1e-3)
    parser.add_argument("--l2", type=float, default=1e-6)
    parser.add_argument("--match", choices=["all", "cls", "patch"], default="all")
    parser.add_argument("--init", choices=["noise", "gray", "zeros"], default="gray")
    parser.add_argument("--restarts", type=int, default=3)
    parser.add_argument(
        "--stages",
        default="1,2,3,4,8,last",
        help="Inversion stages, comma-separated, e.g. 1,2,3,4,8,last",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load backbone (same as main.py, without classifier)
    print("Loading ViT backbone...")
    backbone = load_backbone(pretrained=True).to(device)
    backbone.eval()
    num_blocks = len(backbone.blocks)
    print(f"ViT block count: {num_blocks}")

    # Load and preprocess image
    transform = get_transform()
    img = Image.open(args.image).convert("RGB")
    x_input01 = transform(img).unsqueeze(0).to(device)
    x_input = normalize_for_model(x_input01)

    stage_blocks = parse_stage_blocks(args.stages, num_blocks)
    print(f"Stages to run: {stage_blocks}")

    # Extract target features for each stage
    print("Extracting target features for each stage...")
    target_feats = {}
    with torch.no_grad():
        for block_idx in stage_blocks:
            feat = vit_feature_at_block(backbone, x_input, block_idx)
            target_feats[block_idx] = feat
            print(f"  Block {block_idx} target shape: {feat.shape}")

    print("Starting multi-stage feature inversion...")
    to_pil = transforms.ToPILImage()
    last_saved = None

    for block_idx in stage_blocks:
        print(f"\n[Stage] Block {block_idx}")
        x_recon, best_loss = feature_inversion(
            backbone=backbone,
            target_feat=target_feats[block_idx],
            block_idx=block_idx,
            device=device,
            num_steps=args.steps,
            lr=args.lr,
            feat_weight=args.feat_weight,
            cos_weight=args.cos,
            tv_weight=args.tv,
            l2_weight=args.l2,
            match=args.match,
            init=args.init,
            restarts=args.restarts,
        )
        out_path = f"{args.output_prefix}_block{block_idx}.jpg"
        out_img = x_recon.clamp(0, 1).squeeze(0).cpu()
        to_pil(out_img).save(out_path)
        print(f"  Saved: {out_path} (best loss={best_loss:.6f})")
        last_saved = out_path

    # Legacy: save last stage result to --output path
    if args.output is not None and last_saved is not None:
        Image.open(last_saved).save(args.output)
        print(f"Legacy output saved to: {args.output}")


if __name__ == "__main__":
    main()
