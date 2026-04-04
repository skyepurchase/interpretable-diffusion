import jax
import jax.numpy as jnp
from diffuse.neural_network.nn import CondUNet2D
from flax import nnx

from models.modules.acts import ACTS
from utils.config.models import DDIMConfig


def compute_betas(
    schedule: str, beta_start: float, beta_end: float, num_timesteps: int
) -> jax.Array:
    """
    Produce beta sequence according to provided schedule

    Parameters
    ----------
    schedule : str
        The schedule to create betas
    beta_start : float
        Inclusive beta start value
    beta_end : float
        Inclusive beta end value
    num_timesteps : int
        The number of timesteps used in training
    """
    if schedule == "linear":
        return jnp.linspace(beta_start, beta_end, num_timesteps)

    raise NotImplementedError(f"{schedule} schedule not implemented.")


class DDIM(nnx.Module):
    """
    A wrapper for jcopo/diffuse CondUNet2D

    Parameters
    ----------
    cfg : DDIMConfig
        The configuration for the CondUNet2D
    rngs : nnx.Rngs
        A PRNG key
    """

    def __init__(self, cfg: DDIMConfig, rngs: nnx.Rngs) -> None:
        self.num_timesteps = cfg.num_timesteps
        self.betas = compute_betas(
            cfg.beta_schedule, cfg.beta_start, cfg.beta_end, self.num_timesteps
        )

        self.denoiser = CondUNet2D(
            in_channels=cfg.in_features,
            ch=cfg.base_features,
            ch_mult=cfg.feature_multiples,
            num_res_blocks=cfg.num_blocks,
            attention_resolutions=cfg.attention_resolutions,
            activation=ACTS[cfg.act],
            dropout=cfg.dropout,
            dropout_rate=cfg.dropout_rate,
            num_heads=cfg.attention_heads,
            rngs=rngs,
        )

        self.pad = cfg.pad

    def __call__(self, x: jax.Array, t: jax.Array) -> jax.Array:
        """Wrapper to return the output itself"""
        return self.denoiser(x, t).output
