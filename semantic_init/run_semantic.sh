#!/usr/bin/env bash
# run_semantic.sh — Example commands for semantic warm-start inversion
#
# Step 0: (first time only) generate / download anchor images
#   python prepare_anchors.py
#   python prepare_anchors.py --synthetic-only   # offline / no internet
#
# Step 1: run semantic inversion + gray-init comparison
#   bash run_semantic.sh /path/to/target.jpg
#
# All results land in results/semantic_inversion/

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

IMAGE="${1:?Usage: bash run_semantic.sh <path/to/image.jpg>}"

# ── 1. Prepare anchors (skips if already done) ──────────────────────────────
echo "=== Preparing anchor library ==="
python prepare_anchors.py --anchors-dir anchors/

# ── 2. Semantic warm-start inversion ────────────────────────────────────────
echo ""
echo "=== Running semantic inversion ==="
python semantic_inversion.py \
    --model-name  Qwen/Qwen3.5-VL-3B-Instruct \
    --image       "$IMAGE" \
    --anchors-dir anchors/ \
    --layers      4,8,16,last \
    --steps       1200 \
    --restarts    3 \
    --lr          0.03 \
    --feat-weight 1.0 \
    --cos-weight  0.2 \
    --tv-weight   1e-3 \
    --l2-weight   1e-6 \
    --match       all \
    --compare \
    --output-dir  results/semantic_inversion

echo ""
echo "Results saved in: $SCRIPT_DIR/results/semantic_inversion/"
echo "Open summary.json for per-layer PSNR comparison."
