#!/bin/bash

seeds=(42 9001 27032002 1337 13052002)

for seed in ${seeds[@]}; do
    uv run src/train.py \
        --config configs/diffusion_paid/26MAY06.yaml \
        --seed $seed \
        --local
done
