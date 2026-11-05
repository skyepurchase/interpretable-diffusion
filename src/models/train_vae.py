import jax
import jax.numpy as jnp
from jax.scipy.stats import bernoulli, norm

from models.vae import VAE


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


def vae_loss(model: VAE, x: jax.Array, key: jax.Array) -> tuple[jax.Array, jax.Array]:
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
    loss = -(log_prior.sum() + log_likelihood.sum() - log_posterior.sum())
    return loss, loss[None]
