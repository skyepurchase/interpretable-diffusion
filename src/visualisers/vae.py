import jax
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
    latents = model.reconstruct(images, key=enc_key)

    return model.decode(latents)
