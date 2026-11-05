#!/bin/bash

seeds=(42 9001 27032002 1337 13052002)

for seed in ${seeds[@]}; do
    uv run src/sweep.py \
        --config configs/diffusion_paid/condition_base.yaml \
        --seed $seed \
        --conditions 2 5 10 25 50
done
