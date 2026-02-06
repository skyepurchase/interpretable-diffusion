import jax
import jax.numpy as jnp
from flax import nnx
from tqdm import tqdm

from models.ddim import DDIM


def generate(model: nnx.Module, array_like: jax.Array, key: jax.Array) -> jax.Array:
    """
    Given a trained DDIM model and array who's shape is to be produced generate images.

    Parameters
    ----------
    model : DDIM
        The trained DDIM model
    array_like : jax.Array
        Array who's shape is matched
    key : jax.Array
        PRNG key

    Returns
    images : jax.Array
        The generated images with shape like array_like
    """
    if not isinstance(model, DDIM):
        raise TypeError(f"Expected model to be DDIM but received {type(model)}")

    model.eval()

    img_shape = array_like.shape[1:]
    n_samples = array_like.shape[0]

    steps = jnp.linspace(999, 0, 50, dtype=int)
    alphas = (1 - model.betas).cumprod(axis=0)
    pred_x = array_like.copy()

    for step in tqdm(steps, desc="Sampling"):
        key, noise_key = jax.random.split(key, 2)
        random_start = jax.random.normal(noise_key, shape=(n_samples, *img_shape))

        alpha = alphas[step].reshape(-1, 1, 1, 1)
        x = pred_x * jnp.sqrt(alpha) + random_start * jnp.sqrt(1 - alpha)

        eps = model(x, step * jnp.ones(n_samples))
        pred_x = (x - eps * jnp.sqrt(1.0 - alpha)) / jnp.sqrt(alpha)

    return jnp.clip(pred_x, 0, 1)
