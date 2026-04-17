import jax
import jax.numpy as jnp
import jax.random
from flax import nnx
from jax.random import split

from models.paid import PAID
from models.utils import remove_padding


def loss_fn(
    model: PAID, data: jax.Array, key: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """
    Container function for the separate losses that define the physics aligned interpretable diffusion training.

    Parameters
    ----------
    model : PAID
        The model to train
    data : Array (B, W, H, C)
        The datapoint to calcualte the loss on
    key : PRNG
        Random key

    Returns
    -------
    loss : Array ()
        The total loss of the model on the data
    """
    # TODO: Make condition a config parameter
    condition = 1

    batch_size = data.shape[0]

    key, time_key, noise_key = split(key, num=3)
    noise = jax.random.normal(noise_key, data.shape)

    t = jax.random.randint(
        time_key, (batch_size // 2 + 1,), minval=0, maxval=model.num_timesteps
    )
    t = jnp.concatenate([t, model.num_timesteps - t - 1], axis=0)[:batch_size]

    alpha = (1 - model.betas).cumprod(axis=0)[t].reshape(-1, 1, 1, 1)
    noisy_x = data * jnp.sqrt(alpha) + noise * jnp.sqrt(1.0 - alpha)
    x = jnp.concatenate([data[:, :condition], noisy_x[:, condition:]], axis=1)

    phys_hat, phys, x_hat = model.denoiser.full_forward(
        jnp.pad(x, [(0, 0), model.pad[0], model.pad[1], (0, 0)]), t
    )

    phys_hat = remove_padding(phys_hat, model.pad)
    phys = remove_padding(phys, model.pad)
    x_hat = remove_padding(x_hat, [(0, 0)] + model.pad)

    estimation_loss = (
        jnp.square((data - x_hat[-1])[:, condition:]).sum(axis=(1, 2, 3)).mean()
    )
    phys_loss = (
        jnp.square(phys_hat - jax.lax.stop_gradient(phys)).sum(axis=(1, 2, 3)).mean()
    )

    full_loss = estimation_loss + phys_loss

    return full_loss, jnp.stack([full_loss, estimation_loss, phys_loss])


@nnx.jit
def train_step(
    model: nnx.Module, data: jax.Array, optimizer: nnx.Optimizer, key: jax.Array
) -> jax.Array:
    """
    Carry out a single train step; randomly sample noise and time steps, calculate the model loss, perform gradient descent.

    Parameters
    ----------
    model : PAID
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
    if not isinstance(model, PAID):
        raise TypeError(f"Expected model type to be DDIM not {type(model)}")

    if len(data.shape) < 4:
        data = jnp.expand_dims(data, 3)

    (_, losses), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model, data, key)
    optimizer.update(model, grads)

    return losses
