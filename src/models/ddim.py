import jax
import jax.numpy as jnp
from flax import nnx

from models.modules.acts import ACTS
from utils.config.models import DDIMConfig


def get_beta_schedule(
    schedule: str, beta_start: float, beta_end: float, timesteps: int
) -> jax.Array:
    if schedule == "linear":
        betas = jnp.linspace(beta_start, beta_end, timesteps, dtype=jnp.float32)
    else:
        raise NotImplementedError(f"No implementation for schedule {schedule}")

    assert betas.shape == (timesteps,)
    return betas


def get_timestep_embedding(timesteps: jax.Array, embedding_dim: int) -> jax.Array:
    """
    This matches the implementation in denoising diffusion implicit models (https://github.com/ermongroup/ddim/blob/main/models/diffusion.py), which in turn matches denoising diffusion probabilistic models.
    """
    half_dim = embedding_dim // 2
    emb = jnp.log(10000) / (half_dim - 1)
    emb = jnp.exp(jnp.arange(half_dim, dtype=jnp.float32) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=1)

    if embedding_dim % 2 == 1:
        emb = jnp.pad(emb, ((0, 0), (0, 1), (0, 0), (0, 0)))

    return emb


class DDIM(nnx.Module):
    def __init__(self, cfg: DDIMConfig, denoiser: nnx.Module, rngs: nnx.Rngs) -> None:
        self.in_time_features = cfg.time_embedding
        self.betas = get_beta_schedule(
            cfg.beta_schedule, cfg.beta_start, cfg.beta_end, cfg.num_timesteps
        )
        self.num_timesteps = cfg.num_timesteps

        self.embed_time = nnx.Sequential(
            nnx.Linear(
                in_features=cfg.time_embedding,
                out_features=cfg.time_features,
                rngs=rngs,
            ),
            ACTS[cfg.act],
            nnx.Linear(
                in_features=cfg.time_features, out_features=cfg.time_features, rngs=rngs
            ),
        )

        self.denoiser = denoiser

    def __call__(self, x: jax.Array, t: jax.Array) -> jax.Array:
        time_embedding = get_timestep_embedding(t, self.in_time_features)
        time_embedding = self.embed_time(time_embedding)

        return self.denoiser(x, time_embedding)
