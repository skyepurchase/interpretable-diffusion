from typing import Callable

import jax
import jax.numpy as jnp
from flax import nnx

from models.ddim import DDIM
from models.paid import PAID

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
    Given a trained DDIM style model and array with target shape generate images.

    Parameters
    ----------
    model : DDIM | PAID
        The trained DDIM model
    array_like : jax.Array
        Array who's shape is matched
    key : jax.Array
        PRNG key

    Returns
    images : jax.Array
        The generated images with shape like array_like
    """
    if not isinstance(model, DDIM) and not isinstance(model, PAID):
        raise TypeError(f"Expected model to be DDIM but received {type(model)}")

    model.eval()

    if len(array_like.shape) < 4:
        array_like = jnp.expand_dims(array_like, axis=3)
    array_like = jnp.pad(array_like, ((0, 0), model.pad[0], model.pad[1], (0, 0)))
    img_shape = array_like.shape[1:]
    n_samples = array_like.shape[0]

    steps = jnp.linspace(999, 0, 100, dtype=int)
    alphas = (1 - model.betas).cumprod(axis=0)[steps]
    values = jnp.concatenate([alphas[:, None], steps[:, None]], axis=1)

    init = (model, array_like.copy(), key)

    gen_step = make_step(n_samples, img_shape)
    _, images = jax.lax.scan(gen_step, init, values)

    sample = images[-1]
    if sum(model.pad[0]) > 0:
        sample = sample[:, model.pad[0][0] : -model.pad[0][1]]
    if sum(model.pad[1]) > 0:
        sample = sample[:, :, model.pad[1][0] : -model.pad[1][1]]

    return sample


def generate_mnist(
    model: nnx.Module, array_like: jax.Array, key: jax.Array
) -> jax.Array:
    """
    Wrapper for generating images that clips the output to be in the range 0 and 1.
    """
    return jnp.clip(generate_images(model, array_like, key), 0, 1)


def generate_timeseries(
    model: nnx.Module, array_like: jax.Array, key: jax.Array
) -> jax.Array:
    """
    Wrapper for generating images that are permuted for the correct time and x axes.
    """
    return jnp.permute_dims(generate_images(model, array_like, key), (0, 2, 1, 3))
