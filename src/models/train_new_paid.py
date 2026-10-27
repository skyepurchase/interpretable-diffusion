import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, PRNGKeyArray

from models.new_paid import NewPaid


def coefficient_loss(model: NewPaid, solution: Array, coeffs: list[Array]) -> Array:
    # batch : (B, k*T)
    batch = solution.reshape(solution.shape[0], -1)

    # pred_coeffs : (B, Z)
    pred_coeffs = model.estimator(batch)

    return jnp.square(jnp.concatenate(coeffs, axis=1) - pred_coeffs).mean()


def physics_loss(model: NewPaid, source: Array, target: Array, tau: Array) -> Array:
    # z_tau : (B, k, X)
    z_tau = ((1 - tau.reshape(-1, 1, 1)) * source) + (tau.reshape(-1, 1, 1) * target)

    # pred : (B, k, X)
    pred = model.mapper.step(z_tau, tau)

    return jnp.square(pred - (target - source)).sum((1, 2)).mean()


def reconstruction_loss(model: NewPaid, data: Array) -> Array:
    assert data.shape[1:] == (model.t_grid, model.x_grid), (
        f"Expected data to be a full sample, received {data.shape}."
    )

    # pred : (B, T, X)
    pred = model.solve(data[:, : model.k])

    assert pred.shape == data.shape, (
        f"Expected prediction and data to be same shape, {data.shape} != {pred.shape}"
    )

    return jnp.square(pred - data).sum((1, 2)).mean()


def augmentation_loss(model: NewPaid, data: Array, samples: Array, t: Array) -> Array:
    coeffs = jax.lax.stop_gradient(model.estimate(data[:, : model.k]))

    # y : (B, X)
    y = data[jnp.arange(samples.shape[0]), samples]

    # augmentation : (B, X)
    augmentation = model.augmenter(
        jnp.concatenate(
            [y, coeffs, data[:, : model.k].reshape(-1, model.x_grid * model.k), t],
            axis=1,
        )
    )

    return jnp.square(augmentation).sum(1).mean()


@nnx.jit
def pretrain_step(
    model: NewPaid, data: Array, optimiser: nnx.Optimizer, key: PRNGKeyArray
):
    assert len(data.shape) == 3, (
        f"Expected a sample, full or partial, (B, T, X), received {data.shape}"
    )
    batch_size = data.shape[0]

    coeffs = []
    for coeff_range in model.coeff_ranges:
        key, coeff_key = jax.random.split(key)
        coeffs.append(
            jax.random.uniform(
                coeff_key,
                minval=coeff_range[0],
                maxval=coeff_range[1],
                shape=(batch_size, 1),
            )
        )

    solution = model.gen_physics(data[:, 0], *coeffs)[:, : model.k]

    loss, grads = nnx.value_and_grad(coefficient_loss)(model, solution, coeffs)
    optimiser.update(model, grads)

    return loss[None]


def loss_fn(model: NewPaid, data: Array, key: PRNGKeyArray) -> tuple[Array, Array]:
    assert data.shape[1:] == (model.t_grid, model.x_grid), (
        f"Expected full sample ({model.t_grid}, {model.x_grid}) != {data.shape[1:]}."
    )
    batch_size = data.shape[0]

    # Train g
    coeffs = model.estimate(data[:, : model.k])
    target = model.gen_physics(data[:, 0], coeffs)[:, : model.k]

    key, tau_key = jax.random.split(key)
    tau = jax.random.uniform(tau_key, minval=0.0, maxval=1.0, shape=(batch_size, 1))

    phys_loss = model.phys_reg * physics_loss(
        model, data[:, : model.k], jax.lax.stop_gradient(target), tau
    )

    # Train f
    recon_loss = reconstruction_loss(model, data)

    # Align f and g
    key, sample_key = jax.random.split(key)
    samples = jax.random.randint(
        sample_key,
        minval=0,
        maxval=model.t_grid,
        shape=(batch_size, 1),
        dtype=jnp.int32,
    )
    t = samples * model.dt

    aug_loss = model.aug_reg * augmentation_loss(model, data, samples[:, 0], t)

    full_loss = phys_loss + recon_loss + aug_loss

    return full_loss, jnp.stack([full_loss, recon_loss, phys_loss, aug_loss])


@nnx.jit
def train_step(
    model: NewPaid, optimiser: nnx.Optimizer, data: Array, key: PRNGKeyArray
):
    # Update weights
    (_, losses), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model, data, key)
    optimiser.update(model, grads)

    return losses
