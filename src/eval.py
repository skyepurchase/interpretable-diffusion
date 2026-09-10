import jax.numpy as jnp
from flax import nnx
from jaxtyping import PRNGKeyArray
from torch.utils.data import DataLoader
from tqdm import tqdm

from visualisers.ddim import generate_images


def evaluate(
    model: nnx.Module,
    testloader: DataLoader,
    leave: bool = True,
    *,
    key: PRNGKeyArray,
):
    model.eval()

    mae = jnp.zeros((0, 50, 12))
    for batch, _ in tqdm(testloader, desc="Evaluating", leave=leave):
        preds = generate_images(
            model,
            batch,
            sample_method="ddim",
            estimation_type="mean",
            jump=10,
            key=key,
        )[:, :, :, 0]

        mean_abs_error = jnp.absolute(preds - batch)
        mae = jnp.concatenate([mae, mean_abs_error])

    return mae.mean()
