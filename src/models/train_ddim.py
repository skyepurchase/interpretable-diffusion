import jax
import jax.numpy as jnp
import jax.random
from flax import nnx
from jax.random import split

from models.ddim import DDIM
from models.paid import PAID
from models.utils import remove_padding


def noise_estimation_loss(
    model: DDIM | PAID,
    data: jax.Array,
    t: jax.Array,
    noise: jax.Array,
    betas: jax.Array,
    condition: int = 0,
) -> jax.Array:
    """
    TODO: Meaningful docstring

    Returns
    -------
    loss : Array ()
        The noise estimation loss
    """
    assert condition >= 0, "Condition period must be non-negative"
    # TODO: add more flexibility
    assert condition < data.shape[1], (
        f"Condition period must be less the width of the data, {condition} >= {data.shape[1]}"
    )

    alpha = (1 - betas).cumprod(axis=0)[t].reshape(-1, 1, 1, 1)
    noisy_x = data * jnp.sqrt(alpha) + noise * jnp.sqrt(1.0 - alpha)
    x = jnp.concatenate([data[:, :condition], noisy_x[:, condition:]], axis=1)
    eps = model(jnp.pad(x, [(0, 0), model.pad[0], model.pad[1], (0, 0)]), t)
    eps = remove_padding(eps, model.pad)

    assert eps.shape == noise.shape, (
        f"Noise and estimation must be same shape, {eps.shape} != {noise.shape}."
    )

    return jnp.square((noise - eps)[:, condition:]).sum(axis=(1, 2, 3)).mean(axis=0)


@nnx.jit
def train_step(
    model: nnx.Module, data: jax.Array, optimizer: nnx.Optimizer, key: jax.Array
) -> jax.Array:
    """
    Carry out a single train step; randomly sample noise and time steps, calculate the model loss, perform gradient descent.

    Parameters
    ----------
    model : DDIM
        The model to train
    data : Array
        The single data sample
    optimizer : Optimizer
        The optimizer with the model parameters
    key : Array
        PRNG key
    epoch : int
        The current epoch, used for logging

    Returns
    -------
    losses : Array
        Loss data to be logged
    """
    if not isinstance(model, DDIM):
        raise TypeError(f"Expected model type to be DDIM not {type(model)}")

    if len(data.shape) < 4:
        data = jnp.expand_dims(data, 3)

    batch_size = data.shape[0]
    betas = model.betas

    key, time_key, noise_key = split(key, num=3)
    noise = jax.random.normal(noise_key, data.shape)

    t = jax.random.randint(
        time_key, (batch_size // 2 + 1,), minval=0, maxval=model.num_timesteps
    )
    t = jnp.concatenate([t, model.num_timesteps - t - 1], axis=0)[:batch_size]

    loss, grads = nnx.value_and_grad(noise_estimation_loss)(
        model, data, t, noise, betas
    )
    optimizer.update(model, grads)

    return loss[None]
