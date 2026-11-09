#!/bin/bash

uv run src/sweep.py \
    --config configs/diffusion_paid/condition_base.yaml \
    --conditions 5 10 25 50
