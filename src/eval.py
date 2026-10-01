import jax.numpy as jnp
from flax import nnx
from jaxtyping import PRNGKeyArray
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.paid import PAID
from utils import load_config
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

    forecast_mae = None
    coeff_mae = jnp.zeros((0, 1))
    pbar = tqdm(testloader, desc="Evaluating") if local else testloader
    for batch, coeffs in pbar:
        if len(batch.shape) < 4:
            batch = jnp.expand_dims(batch, axis=-1)

        forecast = generate_images(
            model,
            batch,
            condition=model.condition,
            sample_method="ddim",
            estimation_type="mean",
            jump=10,
            key=key,
        )
        coeffs = jnp.tile(jnp.reshape(coeffs, (1, 64)), reps=(100, 1))
        coeff_mae = jnp.absolute(forecast - coeffs)
        jax.debug.breakpoint()

        assert forecast.shape[1:] == (*batch.shape[1:3], 2), (
            f"Expected forecast to have shape (B, {batch.shape[1]}, {batch.shape[2]}, 2) received {forecast.shape}."
        )
        true_forecast = forecast[:, :, :, :1]
        coeff_pred = forecast[:, 0, 0, 0].reshape(-1, 1)

        if forecast_mae is None:
            forecast_mae = jnp.absolute(true_forecast - batch)
        else:
            forecast_mae = jnp.concatenate(
                [forecast_mae, jnp.absolute(true_forecast - batch)]
            )
        coeff_mae = jnp.concatenate([coeff_mae, jnp.absolute(coeff_pred - coeffs)])

    assert forecast_mae is not None  # typing
    return forecast_mae.mean().item(), coeff_mae.mean().item()


if __name__ == "__main__":
    import os
    from argparse import ArgumentParser

    import jax

    from datasets import DATASETS
    from models import load_model

    parser = ArgumentParser()
    parser.add_argument("-c", "--config", help="path to config file", required=True)
    parser.add_argument(
        "--checkpoint", help="checkpoint file in YYMMDD_HHMMSS format", default="latest"
    )
    parser.add_argument(
        "-s", "--seed", type=int, help="the seed for the run", default=0
    )
    parser.add_argument("--local", action="store_true")

    args = parser.parse_args()
    config = load_config(args.config)

    key = jax.random.key(args.seed)
    rngs = nnx.Rngs(args.seed)

    model = load_model(
        config["model"],
        config["model_config"],
        os.path.join(os.getcwd(), config["checkpoint_dir"], args.checkpoint),
        rngs=rngs,
    )

    data_key, key = jax.random.split(key)
    _, test_loader, _ = DATASETS[config["dataset"]](
        **config["dataset_config"], key=data_key
    )

    forecast, coeff = evaluate(model, test_loader, args.local, key=key)
    print()
    print(f"\nForecast error: {forecast:.2E}, coefficient error: {coeff:.2E}")
