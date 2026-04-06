#!/bin/bash
cd "$(dirname "$0")/.." || exit 1

python qwen_vision_feature_inversion.py \
  --model-name Qwen/Qwen3.5-VL-3B-Instruct \
  --image test.jpg \
  --layers 1,4,8,last \
  --steps 1200 \
  --lr 0.03 \
  --feat-weight 1.0 \
  --cos-weight 0.2 \
  --tv-weight 1e-3 \
  --l2-weight 1e-6 \
  --match all \
  --restarts 2 \
  --output-dir results/qwen_vision_inversion
