from typing import Callable

import jax
import jax.numpy as jnp
from flax import nnx

from models.ddim import DDIM

Carry = tuple[nnx.Module, jax.Array, jax.Array]


def make_step(
    n_samples: int, shape: tuple[int, ...]
) -> Callable[[Carry, jax.Array], tuple[Carry, jax.Array]]:
    """
    Create the generation step keeping the number of samples and shape constant.

    Parameters
    ----------
    n_samples : int
        Number of image samples
    shape : (int, int, int)
        The shape of the images

    Returns
    -------
    step
        Function to be used in jax scan
    """

    def step(carry: Carry, value: jax.Array) -> tuple[Carry, jax.Array]:
        model, prev, key = carry
        alpha_raw, step = jnp.split(value, 2, axis=0)

        key, noise_key = jax.random.split(key, 2)
        noise = jax.random.normal(noise_key, shape=(n_samples, *shape))

        alpha = alpha_raw.reshape(-1, 1, 1, 1)
        x = prev * jnp.sqrt(alpha) + noise * jnp.sqrt(1 - alpha)

        eps = model(x, step * jnp.ones(n_samples))
        curr = (x - eps * jnp.sqrt(1.0 - alpha)) / jnp.sqrt(alpha)

        return (model, curr, key), curr

    return step


def generate_images(
    model: nnx.Module, array_like: jax.Array, key: jax.Array
) -> jax.Array:
    """
    Given a trained DDIM model and array who's shape is to be produced generate images.

    Parameters
    ----------
    model : DDIM
        The trained DDIM model
    array_like : jax.Array
        Array who's shape is matched
    key : jax.Array
        PRNG key

    Returns
    images : jax.Array
        The generated images with shape like array_like
    """
    if not isinstance(model, DDIM):
        raise TypeError(f"Expected model to be DDIM but received {type(model)}")

    model.eval()

    img_shape = array_like.shape[1:]
    n_samples = array_like.shape[0]

    steps = jnp.linspace(999, 0, 100, dtype=int)
    alphas = (1 - model.betas).cumprod(axis=0)[steps]
    values = jnp.concatenate([alphas[:, None], steps[:, None]], axis=1)

    init = (model, array_like.copy(), key)

    gen_step = make_step(n_samples, img_shape)
    _, images = jax.lax.scan(gen_step, init, values)

    return jnp.clip(images[-1], 0, 1)
