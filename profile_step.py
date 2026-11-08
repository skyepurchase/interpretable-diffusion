"""Throwaway profiling script: times train/val steps and their components."""

import sys
import time
from functools import partial
from pathlib import Path

sys.path.insert(0, "src")

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from jax.random import split

from datasets import DATASETS
from models import LOSS_FNS, MODELS
from utils import load_config

config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/pendulum_paid/base.yaml"
config = load_config(config_path)
print(f"Config: {config_path}")

key = jax.random.key(0)
data_key, key = split(key, num=2)
train_loader, val_loader, _ = DATASETS[config["dataset"]](
    **config["dataset_config"], key=data_key
)
model = MODELS[config["model"]](
    **config["model_config"], dataset_config=config["dataset_config"], rngs=nnx.Rngs(0)
)
window = model.window
print(f"window={window} pde_dim={model.pde_dim} dt={model.dt} t1={model.t1}")

loss_fn = LOSS_FNS[config["model"]]

batch, _ = next(iter(train_loader))
batch = batch[:, :window]
print(f"train batch: {batch.shape}, batches/epoch: {len(train_loader)}, val batches: {len(val_loader)}")

wd = config["trainer_config"].get("weight_decay", 0.0)
lr = config["trainer_config"]["learning_rate"]
eps = config["trainer_config"].get("eps", 1e-8)
gt = optax.chain(optax.add_decayed_weights(wd), optax.adam(lr, eps=eps))
clip = config["trainer_config"].get("grad_clip", None)
if clip is not None:
    gt = optax.chain(optax.clip(clip), gt)
optimizer = nnx.Optimizer(model, gt, wrt=nnx.Param)


@nnx.jit
def train_step(model, data, optimizer, key):
    (_, losses), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model, data, key)
    optimizer.update(model, grads)
    return losses


@nnx.jit
def val_step(model, data, key):
    return loss_fn(model, data, key, compute_aug=False)


@nnx.jit
def loss_fwd(model, data, key):
    return loss_fn(model, data, key)[0]


# --- component timings (fixed tau, representative shapes) ---
b = batch.shape[0]
tau = jax.random.randint(key, (b,), minval=0, maxval=model.num_timesteps)


@nnx.jit
def full_forward_only(model, x, tau):
    return model.denoiser.full_forward(x, tau)[0]


@nnx.jit
def generate_physics_only(model, params, init):
    return model.generate_physics(params, init)


def timeit(fn, n=10, **kw):
    fn(**kw)  # compile
    jax.block_until_ready(0)
    t0 = time.perf_counter()
    for _ in range(n):
        out = fn(**kw)
    jax.block_until_ready(out if isinstance(out, jax.Array) else 0)
    return (time.perf_counter() - t0) / n


k = jax.random.key(1)

# compile everything first
t0 = time.perf_counter()
train_step(model=model, data=batch, optimizer=optimizer, key=k).block_until_ready()
print(f"train_step compile: {time.perf_counter() - t0:.1f}s")

t0 = time.perf_counter()
val_step(model=model, data=batch, key=k)[1].block_until_ready()
print(f"val_step compile: {time.perf_counter() - t0:.1f}s")

t0 = time.perf_counter()
loss_fwd(model=model, data=batch, key=k).block_until_ready()
print(f"loss_fwd compile: {time.perf_counter() - t0:.1f}s")

x4 = batch
if x4.ndim < 4:
    x4 = jnp.expand_dims(x4, -1)
t0 = time.perf_counter()
full_forward_only(model=model, x=x4, tau=tau).block_until_ready()
print(f"full_forward compile: {time.perf_counter() - t0:.1f}s")

init = batch[:, 0].reshape(-1, model.num_grid_points)
phys_params = [jax.random.uniform(key, (b, 1)) for _ in model.phys_param_ranges]
t0 = time.perf_counter()
generate_physics_only(model=model, params=phys_params, init=init).block_until_ready()
print(f"generate_physics compile: {time.perf_counter() - t0:.1f}s")

# steady-state timings
ts = timeit(lambda **kw: train_step(**kw), n=10, model=model, data=batch, optimizer=optimizer, key=k)
vs = timeit(lambda **kw: val_step(**kw)[1], n=10, model=model, data=batch, key=k)
fs = timeit(lambda **kw: loss_fwd(**kw), n=10, model=model, data=batch, key=k)
ff = timeit(lambda **kw: full_forward_only(**kw), n=10, model=model, x=x4, tau=tau)
gp = timeit(lambda **kw: generate_physics_only(**kw), n=10, model=model, params=phys_params, init=init)

print(f"\ntrain_step (fwd+bwd+update): {ts*1000:.1f} ms")
print(f"loss_fwd (full loss, no grad): {fs*1000:.1f} ms")
print(f"val_step (no aug solve):       {vs*1000:.1f} ms")
print(f"full_forward (denoiser solve): {ff*1000:.1f} ms")
print(f"generate_physics (B batch):    {gp*1000:.1f} ms")

epochs = config["trainer_config"]["epochs"]
n_train, n_val = len(train_loader), len(val_loader)
per_epoch = ts * n_train + vs * n_val
print(f"\nper-epoch estimate: {per_epoch:.1f} s  ({ts*1000:.0f}ms x {n_train} train + {vs*1000:.0f}ms x {n_val} val)")
print(f"full-run estimate ({epochs} epochs): {per_epoch * epochs / 3600:.2f} hours")
