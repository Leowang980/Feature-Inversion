#!/usr/bin/env bash
# run_semantic.sh — Example commands for semantic warm-start inversion
#
# Step 0: (first time only) anchor images — OpenRouter API, or: python prepare_anchors.py --synthetic-only
#   export OPENROUTER_API_KEY=... && python prepare_anchors.py
#   python prepare_anchors.py --synthetic-only   # no API key / offline
#
# Step 1: run semantic inversion
#   bash run_semantic.sh /path/to/target.jpg
#
# Default: outputs under results/runs/<YYYYMMDD-HHMMSS>/ (no overwrite). Set SEMANTIC_OUTPUT_DIR to pin a folder.
# Default: no gray baseline. To compare: SEMANTIC_COMPARE=1 bash run_semantic.sh ...
#
# Optional overrides:
#   SEMANTIC_STEPS=50 SEMANTIC_LAYERS=1 SEMANTIC_RESTARTS=1 \\
#     bash run_semantic.sh /path/to/target.jpg
#   SEMANTIC_METHOD=best|mixup|knn-mixup  (default: knn-mixup) — change this alone to switch experiments
#   SEMANTIC_KNN_K=5 SEMANTIC_KNN_SIM=neg_l2_mean

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Use workspace .../models/hub when present (same layout as huggingface-cli cache).
if [[ -z "${HF_HUB_CACHE:-}" && -z "${HUGGINGFACE_HUB_CACHE:-}" ]]; then
  for _base in "$SCRIPT_DIR" "$(cd "$SCRIPT_DIR/.." && pwd)" "$(cd "$SCRIPT_DIR/../.." && pwd)"; do
    if [[ -d "$_base/models/hub" ]]; then
      export HF_HUB_CACHE="$_base/models/hub"
      echo "Using local Hugging Face hub cache: $HF_HUB_CACHE"
      # Avoid HEAD/metadata requests to huggingface.co when disk cache is complete.
      export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
      echo "HF_HUB_OFFLINE=${HF_HUB_OFFLINE} (set HF_HUB_OFFLINE=0 to allow network)"
      break
    fi
  done
fi

IMAGE="${1:?Usage: bash run_semantic.sh <path/to/image.jpg>}"

SEMANTIC_STEPS="${SEMANTIC_STEPS:-1200}"
SEMANTIC_LAYERS="${SEMANTIC_LAYERS:-1,4,8,16,last}"
SEMANTIC_RESTARTS="${SEMANTIC_RESTARTS:-3}"
SEMANTIC_METHOD="${SEMANTIC_METHOD:-knn-mixup}"
SEMANTIC_KNN_K="${SEMANTIC_KNN_K:-3}"
SEMANTIC_KNN_SIM="${SEMANTIC_KNN_SIM:-cosine_mean}"
EXTRA_COMPARE=()
if [[ "${SEMANTIC_COMPARE:-0}" == "1" ]]; then
  EXTRA_COMPARE=(--compare)
fi
OUTPUT_ARGS=()
if [[ -n "${SEMANTIC_OUTPUT_DIR:-}" ]]; then
  OUTPUT_ARGS=(--output-dir "${SEMANTIC_OUTPUT_DIR}")
fi

# ── 1. Prepare anchors (optional; skip if anchors/ is ready) ────────────────
# echo "=== Preparing anchor library ==="
# python prepare_anchors.py --anchors-dir anchors/

# ── 2. Semantic warm-start inversion ────────────────────────────────────────
echo ""
echo "=== Running semantic inversion ==="
python semantic_inversion.py \
    --model-name  Qwen/Qwen3.5-4B \
    --image       "$IMAGE" \
    --anchors-dir anchors/ \
    --layers      "${SEMANTIC_LAYERS}" \
    --steps       "${SEMANTIC_STEPS}" \
    --restarts    "${SEMANTIC_RESTARTS}" \
    --lr          0.03 \
    --feat-weight 1.0 \
    --cos-weight  0.2 \
    --tv-weight   1e-3 \
    --l2-weight   1e-6 \
    --match       all \
    --method      "${SEMANTIC_METHOD}" \
    --knn-k       "${SEMANTIC_KNN_K}" \
    --knn-sim     "${SEMANTIC_KNN_SIM}" \
    "${EXTRA_COMPARE[@]}" \
    "${OUTPUT_ARGS[@]}"

echo ""
if [[ -n "${SEMANTIC_OUTPUT_DIR:-}" ]]; then
  echo "Results saved in: $(cd "$SCRIPT_DIR" && realpath "${SEMANTIC_OUTPUT_DIR}")/"
else
  echo "Results saved under: $SCRIPT_DIR/results/runs/<timestamp>/ (see log line Experiment output directory)"
fi
echo "See summary.json (experiment_description) for this run's settings."
