import jax
import jax.numpy as jnp
from flax import nnx

from models.vae import VAE


def reconstruct(model: nnx.Module, images: jax.Array, key: jax.Array) -> jax.Array:
    """
    Given a trained VAE model and set of images produce the reconstructions after encoding and decoding the images.

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

    p_x = model.decode(latents)

    if model.img_gen:
        return p_x
    else:
        mu_x, log_sigma_x = jnp.split(p_x, 2, axis=1)

        key, dec_key = jax.random.split(key)
        Z = jax.random.normal(key=dec_key, shape=mu_x.shape)

        return mu_x + Z * jnp.exp(log_sigma_x)
