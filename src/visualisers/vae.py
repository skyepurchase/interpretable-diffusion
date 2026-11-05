import jax
import jax.numpy as jnp
from flax import nnx

from models.phys_vae import PhysVAE


def reconstruct_point_series(
    model: nnx.Module, series: jax.Array, key: jax.Array
) -> jax.Array:
    """
    Given a physics integrated VAE trained on point-valued time series data and sample series produce reconstructions of the series.

    Parameters
    ----------
    model : PhysVAE
        The trained VAE model
    series : jax.Array
        The time series to reconstruct
    key : jax.Array
        PRNG key

    Returns
    -------
    reconstructions : jax.Array
        The reconstructed series
    """
    if not isinstance(model, PhysVAE):
        raise TypeError(
            f"Expected model to be a physices integrated VAE but received {type(model)}."
        )

    model.eval()

    key, sub_key = jax.random.split(key)

    phys_stats, aux_stats, _ = model.encode(series)
    phys_latents, aux_latents = model.sample_latents(phys_stats, aux_stats, sub_key)
    init_y = series[:, 0].reshape(-1, 1)

    return model.decode(phys_latents, aux_latents, init_y)[0]


def reconstruct_1D_series(model: nnx.Module, series: jax.Array, key: jax.Array):
    """
    Given a physics integrated VAE trained on 1D time series data and sample series produce reconstructions of the series.

    Parameters
    ----------
    model : VAE
        The trained VAE model
    series : jax.Array
        The time series to reconstruct
    key : jax.Array
        PRNG key

    Returns
    -------
    reconstructions : jax.Array
        The reconstructed series
    """
    if not isinstance(model, PhysVAE):
        raise TypeError(f"Expected model to be a VAE but received {type(model)}.")

    model.eval()

    key, enc_key = jax.random.split(key)

    phys_stats, aux_stats, _ = model.encode(series)
    phys_latents, aux_latents = model.sample_latents(phys_stats, aux_stats, enc_key)
    init_y = series[:, : model.num_grid_points]

    # TODO: make this dynamic
    return jnp.permute_dims(
        model.decode(phys_latents, aux_latents, init_y)[0].reshape(-1, 50, 12, 1),
        (0, 2, 1, 3),
    )
