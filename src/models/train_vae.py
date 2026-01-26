import jax
import optax
from flax import nnx
from jax.random import split
from jax.scipy.stats import bernoulli, norm
from torch.utils.data import DataLoader

from models.vae import VAE
from utils.config.trainer import TrainerConfig


def ELBO_loss(model: VAE, x: jax.Array, key: jax.Array) -> jax.Array:
    # Encode batch
    mu_z, sigma_z = model.encode(x)

    # Generate a batched samples
    key, subkey = jax.random.split(key)
    Z = jax.random.normal(key=subkey, shape=mu_z.shape)
    sample = mu_z + Z * sigma_z

    # Decode samples (prevent 0 and 1 absolute probabilities)
    p_x = model.decode(sample)

    # Calculate log probabilities
    log_prior = norm.logpdf(sample)
    log_posterior = norm.logpdf(sample, loc=mu_z, scale=sigma_z)
    log_likelihood = bernoulli.logpmf(x, p=p_x)

    # Assert correct shapes
    assert log_prior.shape == sample.shape, f"{log_prior.shape}"
    assert log_posterior.shape == sample.shape, f"{log_posterior.shape}"
    assert log_likelihood.shape == x.shape, f"{log_likelihood.shape}"

    # Calculate negative ELBO
    return -(
        log_prior.sum(1) + log_likelihood.sum((1, 2, 3)) - log_posterior.sum(1)
    ).sum()


@nnx.jit
def train_step(model: VAE, data: jax.Array, optimizer: nnx.Optimizer, seed: jax.Array):
    loss, grads = nnx.value_and_grad(ELBO_loss)(model, data, seed)
    optimizer.update(model, grads)
    return loss


def train_vae(
    model: nnx.Module, dataloader: DataLoader, cfg: TrainerConfig, key: jax.Array
):
    assert isinstance(model, VAE)

    optimizer = nnx.Optimizer(model, optax.adam(cfg.lr), wrt=nnx.Param)

    for i in range(cfg.epochs):
        running_loss = 0
        for imgs, _ in dataloader:
            key, train_key = split(key, num=2)
            loss = train_step(model, imgs, optimizer, seed=train_key)

            running_loss += loss * imgs.shape[0] / (len(dataloader) * imgs.shape[0])

        print(f"Epoch {i + 1}: loss={running_loss:.2f}")
