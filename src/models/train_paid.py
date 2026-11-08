import jax
import jax.numpy as jnp
import jax.random
from jax.random import split
from jaxtyping import Array

from models.paid import PAID
from models.phys_vae import gen_mask


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


def paid_loss(
    model: PAID, data: Array, key: Array, compute_aug: bool = True
) -> tuple[Array, Array]:
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
    compute_aug : bool = True
        Whether to compute the augmentation loss. Skipping it also skips its
        ODE solve; the returned total is then the reconstruction + physics +
        discrepancy subtotal.

    Returns
    -------
    loss : Array ()
        The total loss of the model on the data
    aux : Array (n)
        The stacked sub-losses of the model on the data
    """
    if not isinstance(model, PAID):
        raise TypeError(f"Expected model type to be PAID not {type(model)}")

    if len(data.shape) < 4:
        data = jnp.expand_dims(data, -1)

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

    # (m, B, T, X, 1), (B, T, X, 1), (B, Z)
    x_hat, phys_pred, phys_params = model.denoiser.full_forward(x, tau)

    reconstruction_loss = (
        # Only the noised generation; the condition frames are left out of the loss
        jnp.square(data[:, model.condition :] - x_hat[-1][:, model.condition :])
        .sum(axis=tuple(range(1, data.ndim)))
        .mean()
    )

    init = data[:, 0].reshape(-1, model.num_grid_points)

    if compute_aug:
        # Sample the augmentation parameters; the key stream matches the
        # original standalone augmentation solve exactly
        key, aug_key = jax.random.split(key, 2)
        aug_params = []
        for param, param_range in zip(phys_params, model.phys_param_ranges):
            aug_key, sample_key = jax.random.split(aug_key, 2)
            aug_params.append(
                jax.lax.stop_gradient(
                    jax.random.uniform(
                        sample_key,
                        param.shape,
                        minval=param_range[0],
                        maxval=param_range[1],
                    )
                )
            )

        # One solve covers both the physics loss and the augmentation loss:
        # the two trajectories only differ in parameters and initial
        # condition, so they are concatenated on the batch axis and split
        # afterwards
        merged_params = [
            jnp.concatenate([param, aug_param], axis=0)
            for param, aug_param in zip(phys_params, aug_params)
        ]
        merged_init = jnp.concatenate([init, init], axis=0)
        merged_traj = model.generate_physics(merged_params, merged_init)
        phys_traj = merged_traj[:batch_size]
    else:
        phys_traj = model.generate_physics(phys_params, init)

    phys_traj = jnp.reshape(phys_traj, phys_pred.shape)
    phys_error = jnp.square(phys_pred - jax.lax.stop_gradient(phys_traj)).sum(axis=-1)
    phys_loss = (
        model.phys_reg * phys_error.sum(axis=tuple(range(1, data.ndim - 1))).mean()
    )

    disc_loss = model.disc_reg * discrepancy_loss(x_hat)

    if compute_aug:
        # (B, X*T)
        aug_phys_hat = jax.lax.stop_gradient(merged_traj[batch_size:])
        aug_loss = jnp.zeros(())
        for phys_func, aug_param in zip(model.denoiser.phys_funcs, aug_params):
            aug_loss += (
                jnp.square(phys_func(aug_phys_hat) - aug_param).sum(axis=1).mean()
            )

        aug_loss = model.aug_reg * aug_loss
        full_loss = reconstruction_loss + phys_loss + disc_loss + aug_loss

        return full_loss, jnp.stack(
            [full_loss, reconstruction_loss, phys_loss, disc_loss, aug_loss]
        )

    # Validation path: the augmentation term and its ODE solve are skipped, so
    # the total is the reconstruction + physics + discrepancy subtotal
    full_loss = reconstruction_loss + phys_loss + disc_loss

    return full_loss, jnp.stack([full_loss, reconstruction_loss, phys_loss, disc_loss])
