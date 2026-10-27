import jax
import jax.numpy as jnp
from diffuse.neural_network.nn import CondUNet2D
from flax import nnx

from models.modules.acts import ACTS


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

    def __init__(
        self,
        num_timesteps: int,
        in_features: int,
        base_features: int,
        feature_multiples: tuple[int, ...],
        attention_resolutions: tuple[int, ...],
        attention_heads: int,
        num_blocks: int = 2,
        act: str = "swish",
        dropout: bool = True,
        dropout_rate: float = 0.1,
        beta_schedule: str = "linear",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        pad: list[tuple] = [(0, 0), (0, 0)],
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.num_timesteps = num_timesteps
        self.betas = compute_betas(
            beta_schedule, beta_start, beta_end, self.num_timesteps
        )

        self.denoiser = CondUNet2D(
            in_channels=in_features,
            ch=base_features,
            ch_mult=feature_multiples,
            num_res_blocks=num_blocks,
            attention_resolutions=attention_resolutions,
            activation=ACTS[act],
            dropout=dropout,
            dropout_rate=dropout_rate,
            num_heads=attention_heads,
            rngs=rngs,
        )

        self.pad = pad

    def __call__(self, x: jax.Array, t: jax.Array) -> jax.Array:
        """Wrapper to return the output itself"""
        return self.denoiser(x, t).output
