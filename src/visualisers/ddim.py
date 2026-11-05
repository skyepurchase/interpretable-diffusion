from typing import Callable, Optional

import jax
import jax.numpy as jnp
from flax import nnx

from models.ddim import DDIM
from models.paid import PAID

Carry = tuple[jax.Array, jax.Array]


def make_ddpm_step(
    n_samples: int,
    model: DDIM | PAID,
    alphas: jax.Array,
    condition: int = 0,
    estimation_type: str = "variance",
) -> Callable[[Carry, jax.Array], tuple[Carry, jax.Array]]:
    """
    Create the generation step keeping the number of samples and shape constant.

    Parameters
    ----------
    n_samples : int
        Number of image samples
    alphas : Array (S)
        The DDIM or PAID alpha values used to train the model
    model : DDIM | PAID
        The model used for generation
    condition : int
        How much of the data is fixed, i.e. conditioned on
    estimation_type : str ("variance", "mean")
        The type of estimation the model makes

    Returns
    -------
    step
        Function to be used in jax scan
    """

    if estimation_type not in ["variance", "mean"]:
        raise NotImplementedError(f"Estimation type {estimation_type} not implemented.")

    assert condition >= 0, "Condition period must be positive"

    def step(carry: Carry, tau: jax.Array) -> tuple[Carry, jax.Array]:
        # x_t[:, :condition] must be the initial condition regardless of t
        x_t, key = carry

        assert condition < x_t.shape[1], (
            f"Condition period must be shorter than provided data, {condition} >= {x_t.shape[1]}"
        )

        t = tau * jnp.ones(n_samples, dtype="int32")

        target_ndim = len(x_t.shape)
        shape = (-1,) + (1,) * (target_ndim - 1)
        alpha_t = alphas[t].reshape(shape)
        alpha_tm1 = alphas[t - 1].reshape(shape)

        pred = model(x_t, t)

        if estimation_type == "variance":
            x_0 = (x_t - pred * jnp.sqrt(1 - alpha_t)) / jnp.sqrt(alpha_t)
        else:
            x_0 = pred

        key, noise_key = jax.random.split(key, 2)
        eps_tm1 = jax.random.normal(noise_key, shape=x_t.shape)

        x_tm1 = jax.lax.cond(
            tau == 1,
            lambda: x_0,
            lambda: x_0 * jnp.sqrt(alpha_tm1) + eps_tm1 * jnp.sqrt(1 - alpha_tm1),
        )

        # Assumes that x_T[:, :condition] is not random noise to begin with
        x_tm1 = jnp.concatenate([x_t[:, :condition], x_tm1[:, condition:]], axis=1)

        return (x_tm1, key), x_tm1

    return step


def make_ddim_step(
    num_samples: int,
    alphas: jax.Array,
    model: PAID,
    condition: int = 5,
    estimation_type: str = "variance",
) -> Callable[[Carry, jax.Array], tuple[Carry, jax.Array]]:
    """
    Create the DDIM generation step function to be used in jax.lax.scan

    Parameters
    ----------
    n_samples : int
        Number of image samples
    alphas : Array (S)
        The DDIM or PAID alpha values used to train the model
    model : DDIM | PAID
        The model used for generation
    condition : int
        How much of the data is fixed, i.e. conditioned on
    estimation_type : str ("variance", "mean")
        The type of estimation the model makes

    Returns
    -------
    step
        Function to be used in jax scan
    """
    if estimation_type not in ["mean", "variance"]:
        raise NotImplementedError(
            f"Estimation type {estimation_type} is not supported."
        )

    def step(carry: Carry, tau_next: jax.Array):
        # x_t[:, :condition] must be the initial condition regardless of t
        x_t, tau = carry

        assert condition < x_t.shape[1], (
            f"Condition period must be shorter than provided data, {condition} >= {x_t.shape[1]}"
        )

        t = tau * jnp.ones(num_samples, dtype="int32")
        t_next = tau_next * jnp.ones(num_samples, dtype="int32")

        target_ndim = len(x_t.shape)
        shape = (-1,) + (1,) * (target_ndim - 1)
        alpha_t = alphas[t].reshape(shape)
        alpha_tm1 = alphas[t_next].reshape(shape)

        x_hat, _, phys_params = model.denoiser.full_forward(x_t, t)
        pred = x_hat[-1]
        # (B, 1)
        coeff = phys_params[0]

        if estimation_type == "variance":
            eps = pred
            x_0 = (x_t - pred * jnp.sqrt(1 - alpha_t)) / jnp.sqrt(alpha_t)
        else:  # Must be "mean" given guard earlier
            x_0 = pred
            eps = (x_t - x_0 * jnp.sqrt(alpha_t)) / jnp.sqrt(1 - alpha_t)

        # x_tm1 : (B, T, X, 1)
        x_tm1 = jax.lax.cond(
            tau_next == 0,
            lambda: x_0,
            lambda: x_0 * jnp.sqrt(alpha_tm1) + eps * jnp.sqrt(1 - alpha_tm1),
        )

        # Assumes that x_T[:, :condition] is not random noise to begin with
        x_tm1 = jnp.concatenate([x_t[:, :condition], x_tm1[:, condition:]], axis=1)

        # out : (B, T, X, 2)
        out = jnp.concatenate(
            [
                x_tm1,
                jnp.tile(
                    jnp.reshape(coeff, (-1, 1, 1, 1)),
                    reps=(1, x_tm1.shape[1], x_tm1.shape[2], 1),
                ),
            ],
            axis=-1,
        )

        return (x_tm1, tau_next), out

    return step


def generate_images(
    model: nnx.Module,
    input: jax.Array,
    condition: int = 0,
    estimation_type: str = "variance",
    sample_method: str = "ddpm",
    jump: Optional[int] = None,
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
    estimation_type : str ("variance", "mean")
        The type of estimation the model makes
    sample_method : str ("ddim", "ddpm")
    key : jax.Array
        PRNG key

    Returns
    images : jax.Array
        The generated images with shape like array_like
    """
    if not isinstance(model, DDIM) and not isinstance(model, PAID):
        raise TypeError(f"Expected model to be DDIM but received {type(model)}")

    model.eval()

    assert len(input.shape) == 4

    n_samples = input.shape[0]

    alphas = (1 - model.betas).cumprod(axis=0)
    key, noise_key = jax.random.split(key)
    x_T = jnp.concatenate(
        [
            input[:, :condition],
            jax.random.normal(noise_key, input.shape)[:, condition:],
        ],
        axis=1,
    )

    if sample_method == "ddim":
        if jump is None:
            jump = 1

        steps = jnp.append(jnp.arange(0, model.num_timesteps - 1)[-jump::-jump], 0)
        init = (x_T, jnp.array(model.num_timesteps - 1))
        gen_step = make_ddim_step(
            n_samples,
            alphas,
            model,
            condition=condition,
            estimation_type=estimation_type,
        )
    elif sample_method == "ddpm":
        if jump is not None:
            print(
                "WARNING: jump parameter provided but using DDPM. jump will be ignored."
            )

        steps = jnp.arange(1, model.num_timesteps)[::-1]
        init = (x_T, key)
        gen_step = make_ddpm_step(
            n_samples,
            model,
            alphas,
            condition=condition,
            estimation_type=estimation_type,
        )
    else:
        raise NotImplementedError(
            f"Sampling method {sample_method} is not supported. Only 'ddim' and 'ddpm' are supported"
        )

    # images (S, B, T, X, 2)
    _, images = jax.lax.scan(gen_step, init, steps)

    return images[-1]


def generate_timeseries(
    model: nnx.Module, initial_condition: jax.Array, jump: int = 10, *, key: jax.Array
) -> jax.Array:
    """
    Wrapper for generating images that are permuted for the correct time and x axes.
    """
    sample_method = "ddim"

    if isinstance(model, PAID):
        condition = model.condition
        est_type = "mean"
    else:
        condition = 1
        est_type = "variance"

    generated = generate_images(
        model,
        initial_condition,
        condition=condition,
        sample_method=sample_method,
        estimation_type=est_type,
        jump=jump,
        key=key,
    )[:, :, :, :1]

    return jnp.permute_dims(generated, (0, 2, 1, 3))
