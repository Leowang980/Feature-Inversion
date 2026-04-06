#!/bin/bash
cd "$(dirname "$0")/.." || exit 1

python qwen_feature_inversion.py \
  --model-name Qwen/Qwen3.5-0.6B \
  --text "The quick brown fox jumps over the lazy dog." \
  --layers 1,4,8,last \
  --steps 800 \
  --lr 0.05 \
  --cos-weight 0.2 \
  --entropy-weight 1e-3 \
  --restarts 2 \
  --max-length 64 \
  --output-dir results/qwen_inversion
