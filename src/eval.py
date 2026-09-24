import jax.numpy as jnp
from flax import nnx
from jaxtyping import PRNGKeyArray
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.paid import PAID
from visualisers.ddim import generate_images


def evaluate(
    model: nnx.Module,
    testloader: DataLoader,
    local: bool = False,
    *,
    key: PRNGKeyArray,
):
    assert isinstance(model, PAID), f"{type(model)} is not supported."
    model.eval()

    forecast_mae = jnp.zeros((0, 50, 12))
    coeff_mae = jnp.zeros((0, 1))
    pbar = tqdm(testloader, desc="Evaluating") if local else testloader
    for batch, coeffs in pbar:
        forecast = generate_images(
            model,
            batch,
            condition=model.condition,
            sample_method="ddim",
            estimation_type="mean",
            jump=10,
            key=key,
        ).reshape(batch.shape)

        _, _, preds, _ = model.denoiser._embed(forecast, jnp.zeros((batch.shape[0], 1)))

        forecast_mae = jnp.concatenate([forecast_mae, jnp.absolute(forecast - batch)])
        coeff_mae = jnp.concatenate([coeff_mae, jnp.absolute(preds[0] - coeffs)])

    return forecast_mae.mean(), coeff_mae.mean()
