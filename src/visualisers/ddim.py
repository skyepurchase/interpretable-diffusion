from typing import Callable

import jax
import jax.numpy as jnp
from flax import nnx

from models.ddim import DDIM, compute_alphas
from models.paid import PAID
from models.utils import remove_padding

Carry = tuple[DDIM | PAID, jax.Array, jax.Array, jax.Array]
DDPMCarry = tuple[DDIM | PAID, jax.Array, jax.Array]


def make_ddim_step(
    n_samples: int,
    shape: tuple[int, ...],
    condition: int = 0,
    estimation_type: str = "noise",
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
    estimation_type : str ("noise", "clean")
        The type of estimation the model makes

    Returns
    -------
    step
        Function to be used in jax scan
    """
    print("Using DDIM\n")

    if estimation_type not in ["noise", "clean"]:
        raise NotImplementedError(f"Estimation type {estimation_type} not implemented.")

    # noise_est = estimation_type == "noise"

    assert condition >= 0, "Condition period must be positive"

    def step(carry: Carry, value: jax.Array) -> tuple[Carry, jax.Array]:
        model, x_t, alpha_t, key = carry
        alpha_raw, t = jnp.split(value, 2, axis=0)
        alpha_tm = alpha_raw.reshape(-1, 1, 1, 1)

        assert condition < x_t.shape[1], (
            f"Condition period must be shorter than provided data, {condition} >= {x_t.shape[1]}"
        )

        pred_noise = model(
            jnp.pad(x_t, ((0, 0), model.pad[0], model.pad[1], (0, 0))),
            t * jnp.ones(n_samples),
        )
        pred_noise = remove_padding(pred_noise, model.pad)

        x_0_hat = (x_t - (pred_noise * jnp.sqrt(1.0 - alpha_t))) / jnp.sqrt(alpha_t)

        # key, noise_key = jax.random.split(key, 2)
        # noise = jax.random.normal(noise_key, shape=(n_samples, *shape))

        # sigma_t = jnp.sqrt((1 - alpha_tm) / (1 - alpha_t)) * jnp.sqrt(1 - (alpha_t / alpha_tm))

        x_tm = (x_0_hat * jnp.sqrt(alpha_tm)) + (pred_noise * jnp.sqrt(1 - alpha_tm))

        return (model, x_tm, alpha_tm, key), x_tm

    return step


def make_ddpm_step(
    n_samples: int,
    shape: tuple[int, ...],
    condition: int = 0,
    estimation_type: str = "noise",
) -> Callable[[DDPMCarry, jax.Array], tuple[DDPMCarry, jax.Array]]:
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
    estimation_type : str ("noise", "clean")
        The type of estimation the model makes

    Returns
    -------
    step
        Function to be used in jax scan
    """
    print("Using DDPM\n")

    if estimation_type not in ["noise", "clean"]:
        raise NotImplementedError(f"Estimation type {estimation_type} not implemented.")

    noise_est = estimation_type == "noise"

    assert condition >= 0, "Condition period must be positive"

    def step(carry: DDPMCarry, value: jax.Array) -> tuple[DDPMCarry, jax.Array]:
        model, prev, key = carry
        alpha_raw, step = jnp.split(value, 2, axis=0)

        assert condition < prev.shape[1], (
            f"Condition period must be shorter than provided data, {condition} >= {prev.shape[1]}"
        )

        key, noise_key = jax.random.split(key, 2)
        noise = jax.random.normal(noise_key, shape=(n_samples, *shape))

        alpha = alpha_raw.reshape(-1, 1, 1, 1)
        noisy_x = prev * jnp.sqrt(alpha) + noise * jnp.sqrt(1 - alpha)
        x = jnp.concatenate([prev[:, :condition], noisy_x[:, condition:]], axis=1)

        pred_noise = model(
            jnp.pad(x, ((0, 0), model.pad[0], model.pad[1], (0, 0))),
            step * jnp.ones(n_samples),
        )
        pred_noise = remove_padding(pred_noise, model.pad)

        if noise_est:
            denoised = (x - pred_noise * jnp.sqrt(1.0 - alpha)) / jnp.sqrt(alpha)
            curr = jnp.concatenate(
                [prev[:, :condition], denoised[:, condition:]], axis=1
            )
        else:
            curr = jnp.concatenate(
                [prev[:, :condition], pred_noise[:, condition:]], axis=1
            )

        return (model, curr, key), curr

    return step


def generate_images(
    model: nnx.Module,
    array_like: jax.Array,
    condition: int = 0,
    estimation_type: str = "noise",
    *,
    key: jax.Array,
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
    estimation_type : str ("noise", "clean")
        The type of estimation the model makes
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

    steps = jnp.linspace(model.num_timesteps - 1, 0, 100, dtype=int)

    alphas = model.alphas[steps]
    alphas = compute_alphas("cosine", 0, 0, model.num_timesteps)[steps]
    curr_alpha = alphas[0].reshape(-1, 1, 1, 1)
    next_alphas = jnp.concatenate([alphas[1:], jnp.ones(1)], axis=0)

    values = jnp.concatenate([next_alphas[:, None], steps[:, None]], axis=1)

    key, noise_key = jax.random.split(key, 2)
    noise = jax.random.normal(noise_key, shape=(n_samples, *img_shape))

    init = (model, noise, curr_alpha, key)

    gen_step = make_ddim_step(
        n_samples, shape=img_shape, condition=condition, estimation_type=estimation_type
    )
    _, images = jax.lax.scan(gen_step, init, values)

    return images[:, 1]


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
    if isinstance(model, PAID):
        condition = model.condition
        est_type = "clean"
    else:
        condition = 1
        est_type = "noise"

    generated = generate_images(
        model, array_like, condition=condition, estimation_type=est_type, key=key
    )

    return jnp.permute_dims(generated, (0, 2, 1, 3))
