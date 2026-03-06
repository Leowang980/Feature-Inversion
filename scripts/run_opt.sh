#!/bin/bash
cd "$(dirname "$0")/.." || exit 1
python optimization.py \
    --image test.jpg \
    --output opt.jpg \
    --output-prefix opt \
    --stages 1,2,3,4,8,last \
    --steps 1500 \
    --lr 0.005 \
    --tv 1e-2 \
    --l2 1e-6 \
    --match all \
    --init gray \
    --restarts 3