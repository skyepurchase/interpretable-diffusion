import optax
import jax
from jax.random import split
from jax.scipy.stats import norm, bernoulli
from flax import nnx

from .vae import VAE


def ELBO_loss(model: VAE, x: jax.Array, key: jax.Array):
    # Encode batch
    mu_z, sigma_z = model.encode(x)

    # Generate a batched samples
    key, subkey = jax.random.split(key)
    Z = jax.random.normal(key=subkey, shape=mu_z.shape)
    sample = mu_z + Z * sigma_z

    # Decode samples
    p_x = model.decode(sample)

    # Calculate log probabilities
    log_prior = norm.logpdf(sample)
    log_posterior = norm.logpdf(
        sample, loc=mu_z, scale=sigma_z)
    log_likelihood = bernoulli.logpmf(x, p=p_x)

    # Assert correct shapes
    assert log_prior.shape == sample.shape, f"{log_prior.shape}"
    assert log_posterior.shape == sample.shape, f"{log_posterior.shape}"
    assert log_likelihood.shape == x.shape, f"{log_likelihood.shape}"

    # Calculate negative ELBO
    return -(log_prior.sum(1) + log_likelihood.sum((1,2,3)) - log_posterior.sum(1)).sum()


@nnx.jit
def train_step(
    model: VAE,
    data: jax.Array,
    optimizer: nnx.Optimizer,
    seed: jax.Array
):
    loss_gradient = nnx.grad(ELBO_loss)
    grads = loss_gradient(model, data, seed)
    optimizer.update(model, grads)


def train_vae(
    model: VAE,
    data: jax.Array,
    epochs: int = 1000,
    lr: float=1e-2,
    *,
    key: jax.Array
):
    optimizer = nnx.Optimizer(
        model, optax.adam(lr), wrt=nnx.Param
    )
    for i in range(epochs+1):
        key, subkey = split(key)
        train_step(model, data, optimizer, seed=subkey)

        if i % 100 == 0:
            key, subkey = split(key)
            loss = ELBO_loss(model, data, key=subkey)
            print(f"Epoch {i}: loss={loss:.2f}")
