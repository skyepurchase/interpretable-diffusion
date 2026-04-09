from typing import Callable

import jax
import jax.numpy as jnp
from flax import nnx

from models.ddim import DDIM
from models.paid import PAID

Carry = tuple[nnx.Module, jax.Array, jax.Array]


def make_step(
    n_samples: int, shape: tuple[int, ...], condition: int = 0
) -> Callable[[Carry, jax.Array], tuple[Carry, jax.Array]]:
    """
    Create the generation step keeping the number of samples and shape constant.

    Parameters
    ----------
    n_samples : int
        Number of image samples
    shape : (int, int, int)
        The shape of the images
    condition : int
        How much of the data is fixed, i.e. conditioned on

    Returns
    -------
    step
        Function to be used in jax scan
    """

    assert condition >= 0, "Condition period must be positive"

    def step(carry: Carry, value: jax.Array) -> tuple[Carry, jax.Array]:
        model, prev, key = carry
        alpha_raw, step = jnp.split(value, 2, axis=0)

        assert isinstance(model, DDIM) or isinstance(model, PAID), (
            f"Only denoising diffusion models accepted, received {type(model)}."
        )
        assert condition < prev.shape[1], (
            f"Condition period must be shorter than provided data, {condition} >= {prev.shape[1]}"
        )

        key, noise_key = jax.random.split(key, 2)
        noise = jax.random.normal(noise_key, shape=(n_samples, *shape))

        alpha = alpha_raw.reshape(-1, 1, 1, 1)
        noisy_x = prev * jnp.sqrt(alpha) + noise * jnp.sqrt(1 - alpha)
        x = jnp.concatenate([prev[:, :condition], noisy_x[:, condition:]], axis=1)

        eps = model(
            jnp.pad(x, ((0, 0), model.pad[0], model.pad[1], (0, 0))),
            step * jnp.ones(n_samples),
        )

        if sum(model.pad[0]) > 0:
            eps = eps[:, model.pad[0][0] : -model.pad[0][1]]
        if sum(model.pad[1]) > 0:
            eps = eps[:, :, model.pad[1][0] : -model.pad[1][1]]

        noisy_curr = (x - eps * jnp.sqrt(1.0 - alpha)) / jnp.sqrt(alpha)
        curr = jnp.concatenate([prev[:, :condition], noisy_curr[:, condition:]], axis=1)

        return (model, curr, key), curr

    return step


def generate_images(
    model: nnx.Module, array_like: jax.Array, condition: int = 0, *, key: jax.Array
) -> jax.Array:
    """
    Given a trained DDIM style model and array with target shape generate images.

    Parameters
    ----------
    model : DDIM | PAID
        The trained DDIM model
    array_like : jax.Array
        Array who's shape is matched
    condition : int
        How much of the data is fixed
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
    img_shape = array_like.shape[1:]
    n_samples = array_like.shape[0]

    steps = jnp.linspace(999, 0, 100, dtype=int)
    alphas = (1 - model.betas).cumprod(axis=0)[steps]
    values = jnp.concatenate([alphas[:, None], steps[:, None]], axis=1)

    init = (model, array_like.copy(), key)

    gen_step = make_step(n_samples, img_shape, condition=condition)
    _, images = jax.lax.scan(gen_step, init, values)

    return images[-1]


def generate_mnist(
    model: nnx.Module, array_like: jax.Array, key: jax.Array
) -> jax.Array:
    """
    Wrapper for generating images that clips the output to be in the range 0 and 1.
    """
    return jnp.clip(generate_images(model, array_like, key=key), 0, 1)


def generate_timeseries(
    model: nnx.Module, array_like: jax.Array, key: jax.Array
) -> jax.Array:
    """
    Wrapper for generating images that are permuted for the correct time and x axes.
    """
    generated = generate_images(model, array_like, condition=1, key=key)
    images = jnp.concatenate(
        [generated[:3], jnp.expand_dims(array_like, axis=3)[:3]], axis=0
    )

    return jnp.permute_dims(images, (0, 2, 1, 3))
