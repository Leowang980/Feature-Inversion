#!/bin/bash
cd "$(dirname "$0")/.." || exit 1
python DP.py \
    --ckpt vit_cifar100_best.pt \
    --stages 1,2,3,4,8,12 \
    --laplace-scale 0.02 \
    --run-acc \
    --run-inversion \
    --image test.jpg \
    --output-dir results/dp_laplace \
    --inv-steps 1500 \
    --inv-lr 0.005 \
    --inv-tv 1e-2 \
    --inv-l2 1e-6 \
    --inv-match all \
    --inv-init gray \
    --inv-restarts 3
