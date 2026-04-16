#!/usr/bin/env bash
# run_semantic.sh — Example commands for semantic warm-start inversion
#
# Step 0: (first time only) anchor images — OpenRouter API, or synthetic fallback
#   export OPENROUTER_API_KEY=... && python prepare_anchors.py
#   python prepare_anchors.py --synthetic-only   # no API key / offline
#
# Step 1: run semantic inversion + gray-init comparison
#   bash run_semantic.sh /path/to/target.jpg
#
# All results land in results/semantic_inversion/

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Use workspace …/models/hub when present (same layout as huggingface-cli cache).
if [[ -z "${HF_HUB_CACHE:-}" && -z "${HUGGINGFACE_HUB_CACHE:-}" ]]; then
  for _base in "$SCRIPT_DIR" "$(cd "$SCRIPT_DIR/.." && pwd)" "$(cd "$SCRIPT_DIR/../.." && pwd)"; do
    if [[ -d "$_base/models/hub" ]]; then
      export HF_HUB_CACHE="$_base/models/hub"
      echo "Using local Hugging Face hub cache: $HF_HUB_CACHE"
      break
    fi
  done
fi

IMAGE="${1:?Usage: bash run_semantic.sh <path/to/image.jpg>}"

# ── 1. Prepare anchors (skips if already done) ──────────────────────────────
echo "=== Preparing anchor library ==="
python prepare_anchors.py --anchors-dir anchors/

# ── 2. Semantic warm-start inversion ────────────────────────────────────────
echo ""
echo "=== Running semantic inversion ==="
python semantic_inversion.py \
    --model-name  Qwen/Qwen3.5-4B \
    --image       "$IMAGE" \
    --anchors-dir anchors/ \
    --layers      1,4,8,16,last \
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
