import jax
import jax.numpy as jnp
from flax import nnx

from models.vae import VAE


def reconstruct_images(
    model: nnx.Module, images: jax.Array, key: jax.Array
) -> jax.Array:
    """
    Given a VAE trained on image data and sample images produce reconstructions of the images.

    Parameters
    ----------
    model : VAE
        The trained VAE model
    images : jax.Array
        The images to reconstruct
    key : jax.Array
        PRNG key

    Returns
    -------
    reconstructions : jax.Array
        The reconstructed images
    """
    if not isinstance(model, VAE):
        raise TypeError(f"Expected model to be a VAE but received {type(model)}.")

    model.eval()

    key, enc_key = jax.random.split(key)
    latents = model.encode_and_sample(images, key=enc_key)

    return model.decode(latents)


def reconstruct_point_series(model: nnx.Module, series: jax.Array, key: jax.Array):
    """
    Given a VAE trained on point-valued time series data and sample series produce reconstructions of the series.

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
    if not isinstance(model, VAE):
        raise TypeError(f"Expected model to be a VAE but received {type(model)}.")

    model.eval()

    key, enc_key = jax.random.split(key)
    latents = model.encode_and_sample(series, key=enc_key)

    return jnp.split(model.decode(latents), 2)[0]


def reconstruct_1D_series(model: nnx.Module, series: jax.Array, key: jax.Array):
    """
    Given a VAE trained on 1D time series data and sample series produce reconstructions of the series.

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
    if not isinstance(model, VAE):
        raise TypeError(f"Expected model to be a VAE but received {type(model)}.")

    model.eval()

    key, enc_key = jax.random.split(key)
    latents = model.encode_and_sample(series, key=enc_key)

    # TODO: make this dynamic
    return jnp.split(model.decode(latents), 2)[0].reshape(-1, 12, 50, 1)
