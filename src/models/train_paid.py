import jax
import jax.numpy as jnp
import jax.random
from flax import nnx
from jax.random import split
from jaxtyping import Array

from models.paid import PAID
from models.phys_vae import gen_mask
from models.utils import remove_padding


def augmentation_loss(
    model: PAID,
    phys_params: list[Array],
    y_init: Array,
    tau: Array,
    key: Array,
) -> Array:
    """
    TODO: write docstring

    Parameters
    ----------
    model : PAID
        Model to train
    phys_params : list[Array (B, Z)]
        Estimated physics parameters
    tau : Array (B)
        Diffusion timestep
    y_init : Array (B, X)
        The initial condition

    Returns
    -------
    loss : Array ()
        The data augmentation loss
    """
    model.eval()

    aug_params = []
    for param, param_range in zip(phys_params, model.phys_param_ranges):
        key, aug_key = jax.random.split(key, 2)
        aug_param = jax.lax.stop_gradient(
            jax.random.uniform(
                aug_key, param.shape, minval=param_range[0], maxval=param_range[1]
            )
        )
        aug_params.append(aug_param)

    aug_phys_hat = jax.lax.stop_gradient(
        model.generate_physics(aug_params, jax.lax.stop_gradient(y_init))
    )

    model.train()

    t_emb = model.denoiser.unet.time_proj(tau)
    t_emb = model.denoiser.unet.time_embedding(t_emb)

    aug_loss = jnp.zeros(())
    for phys_func, aug_param in zip(model.denoiser.phys_funcs, aug_params):
        phys_features = jnp.pad(aug_phys_hat, [(0, 0), *model.pad, (0, 0)])
        for block in model.denoiser.unet.blocks_down:
            phys_features = block(phys_features, t_emb)

        phys_features = model.denoiser.unet.block_mid(phys_features, t_emb)

        aug_loss += jnp.square(phys_func(phys_features) - aug_param).sum(axis=1).mean()

    return aug_loss


def discrepancy_loss(x_hats: Array) -> Array:
    """
    TODO: create docstring

    Parameters
    ----------
    x_hats : Array (m, B, W, H, C)
        The estimated clean sample with different auxiliary functions

    Returns
    -------
    loss : Array ()
        The discrepancy loss
    """
    mask = gen_mask(x_hats.shape[0])

    # discrepancies : (m - 1, m - 1, B)
    discrepancies = jax.vmap(
        # x : (B, W, H, C)
        lambda x: jax.vmap(
            # y : (B, W, H, C)
            lambda y: jnp.square(x - y).sum(tuple(range(1, x.ndim))),
            in_axes=0,
            out_axes=0,
        )(x_hats[:-1]),
        in_axes=0,
        out_axes=0,
    )(x_hats[1:])

    masked_disc = discrepancies.mean(axis=2) * mask

    return masked_disc.sum() / mask.sum()


def loss_fn(model: PAID, data: Array, key: Array) -> tuple[Array, Array]:
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
    batch_size = data.shape[0]

    key, time_key, noise_key = split(key, num=3)
    noise = jax.random.normal(noise_key, data.shape)

    tau = jax.random.randint(
        time_key, (batch_size // 2 + 1,), minval=0, maxval=model.num_timesteps
    )
    tau = jnp.concatenate([tau, model.num_timesteps - tau - 1], axis=0)[:batch_size]

    alpha_shape = (-1,) + (1,) * (data.ndim - 1)
    alpha = (1 - model.betas).cumprod(axis=0)[tau].reshape(*alpha_shape)
    noisy_x = data * jnp.sqrt(alpha) + noise * jnp.sqrt(1.0 - alpha)
    x = jnp.concatenate(
        [data[:, : model.condition], noisy_x[:, model.condition :]], axis=1
    )

    x_hat, phys_pred, phys_params = model.denoiser.full_forward(
        jnp.pad(x, [(0, 0), *model.pad, (0, 0)]), tau
    )

    phys_pred = remove_padding(phys_pred, model.pad)
    x_hat = remove_padding(x_hat, [(0, 0)] + model.pad)
    phys_gen = x_hat[0]

    reconstruction_loss = (
        # Reconstruct the full data to improve smoothness
        jnp.square((data - x_hat[-1])).sum(axis=tuple(range(1, data.ndim))).mean()
    )

    phys_error = jnp.square(phys_pred - jax.lax.stop_gradient(phys_gen)).sum(axis=-1)
    phys_loss = (
        model.phys_reg
        * (
            phys_error[:, : model.condition].sum(axis=tuple(range(1, data.ndim - 1)))
            + (
                jnp.reshape(alpha, alpha_shape[:-1]) * phys_error[:, model.condition :]
            ).sum(axis=tuple(range(1, data.ndim - 1)))
        ).mean()
    )

    disc_loss = model.disc_reg * discrepancy_loss(x_hat)

    key, aug_key = jax.random.split(key, 2)
    aug_loss = model.aug_reg * augmentation_loss(
        model,
        phys_params,
        data[:, 0].reshape(-1, model.num_grid_points),
        tau,
        key=aug_key,
    )

    full_loss = reconstruction_loss + phys_loss + disc_loss + aug_loss

    return full_loss, jnp.stack(
        [full_loss, reconstruction_loss, phys_loss, disc_loss, aug_loss]
    )


@nnx.jit
def train_step(
    model: nnx.Module, data: Array, optimizer: nnx.Optimizer, key: Array
) -> Array:
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
        raise TypeError(f"Expected model type to be PAID not {type(model)}")

    if len(data.shape) < model.dimension + 2:
        data = jnp.expand_dims(data, -1)

    (_, losses), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model, data, key)
    optimizer.update(model, grads)

    return losses
