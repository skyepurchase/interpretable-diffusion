import jax
import jax.numpy as jnp
import jax.random
import optax
from flax import nnx
from jax.random import split
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.ddim import DDIM
from utils.config.trainer import TrainerConfig


def noise_estimation_loss(
    model: DDIM, data: jax.Array, t: jax.Array, noise: jax.Array, betas: jax.Array
) -> jax.Array:
    """
    This implementation matches that in the original DDIM paper https://github.com/ermongroup/ddim/blob/main/functions/losses.py.
    """
    alpha = (1 - betas).cumprod(axis=0)[t].reshape(-1, 1, 1, 1)
    x = data * jnp.sqrt(alpha) + noise * jnp.sqrt(1.0 - alpha)
    output = model(x, t)

    return jnp.square(noise - output).sum(axis=(1, 2, 3)).mean(axis=0)


@nnx.jit
def train_step(
    model: DDIM, data: jax.Array, optimizer: nnx.Optimizer, key: jax.Array
) -> jax.Array:
    batch_size = data.shape[0]
    betas = model.betas

    key, time_key, noise_key = split(key, num=3)
    noise = jax.random.normal(noise_key, data.shape)

    t = jax.random.randint(
        time_key, (batch_size // 2 + 1,), minval=0, maxval=model.num_timesteps
    )
    t = jnp.concatenate([t, model.num_timesteps - t - 1], axis=0)[:batch_size]

    loss, grads = nnx.value_and_grad(noise_estimation_loss)(
        model, data, t, noise, betas
    )
    optimizer.update(model, grads)

    return loss


def train_ddim(
    model: nnx.Module, dataloader: DataLoader, cfg: TrainerConfig, key: jax.Array
):
    assert isinstance(model, DDIM)

    optimizer = nnx.Optimizer(model, optax.adam(cfg.lr), wrt=nnx.Param)

    pbar = tqdm(range(cfg.epochs), desc="Training")
    for _ in pbar:
        running_loss = 0

        for imgs, _ in tqdm(dataloader, desc="Iterating images", leave=False):
            key, train_key = split(key, num=2)
            loss = train_step(model, imgs, optimizer, key=train_key)

            running_loss += loss

        pbar.set_description(f"Training, loss={running_loss:.2f}")
