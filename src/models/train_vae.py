import jax
import jax.numpy as jnp
import optax
from flax import nnx
from jax.random import split
from jax.scipy.stats import bernoulli, norm
from torch.utils.data import DataLoader

from models.vae import VAE
from utils.config.trainer import TrainerConfig


def output_logpmf(dist: str, p_x: jax.Array, x: jax.Array) -> jax.Array:
    """
    For the given distribution and statistics calculate the log probability mass function of x.

    Parameters
    ----------
    dist : str
        The distribution to use
    p_x : Array
        A batched array with statistics for the output shape
    x : (B, shape)
        A sample to calculate the probability of

    Returns
    -------
    logpmf : (B, shape)
        The log probability mass function of x
    """
    if dist == "bernoulli":
        return bernoulli.logpmf(x, p=p_x)
    elif dist == "normal":
        mu_x, log_sigma_x = jnp.split(p_x, 2, axis=1)
        return norm.logpdf(x, loc=mu_x, scale=jnp.exp(log_sigma_x))
    else:
        raise NotImplementedError(f"{dist} distribution not implemented.")


def ELBO_loss(model: VAE, x: jax.Array, key: jax.Array) -> jax.Array:
    """
    Calculates the ELBO for a given model and datapoint.

    Parameters
    ----------
    model : VAE
        The model to evaluate
    x : Array
        The datapoint to evaluate on
    key : Array
        PRNG key

    Returns
    -------
    loss : Array
        The mean ELBO of the model across the datapoints
    """
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
    output_dist = "bernoulli" if model.img_gen else "normal"
    log_likelihood = output_logpmf(output_dist, p_x, x)

    # Assert correct shapes
    assert log_prior.shape == sample.shape, f"{log_prior.shape}"
    assert log_posterior.shape == sample.shape, f"{log_posterior.shape}"
    assert log_likelihood.shape == x.shape, f"{log_likelihood.shape}"

    # Calculate negative ELBO
    return -(log_prior.sum() + log_likelihood.sum() - log_posterior.sum())


@nnx.jit
def train_step(model: VAE, data: jax.Array, optimizer: nnx.Optimizer, key: jax.Array):
    """
    Carry out a single VAE train step

    Parameters
    ----------
    model : VAE
        The model to train
    data : Array
        The datapoint
    optimizer : Optimizer
        The optimizer for the models parameters
    key : Array
        PRNG key
    output_dist : str = bernoulli
        The distribution to use for the output
    """
    loss, grads = nnx.value_and_grad(ELBO_loss)(model, data, key)
    optimizer.update(model, grads)
    return loss


def train_vae(
    model: nnx.Module, dataloader: DataLoader, cfg: TrainerConfig, key: jax.Array
):
    """
    Train a VAE model on the provided dataset.

    Parameters
    ----------
    model : VAE
        The model to train
    dataloader : DataLoader
        The dataloader for the dataset
    cfg : TrainerConfig
        Config specifying epochs and learning rate
    key : Array
        PRNG key
    """
    assert isinstance(model, VAE)

    optimizer = nnx.Optimizer(model, optax.adam(cfg.lr), wrt=nnx.Param)

    for i in range(cfg.epochs):
        running_loss = 0
        for data, _ in dataloader:
            key, train_key = split(key, num=2)
            loss = train_step(model, data, optimizer, key=train_key)

            running_loss += loss * data.shape[0] / (len(dataloader) * data.shape[0])

        print(f"Epoch {i + 1}: loss={running_loss:.2f}")
