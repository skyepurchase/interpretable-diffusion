import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import PRNGKeyArray
from tqdm import tqdm

from datasets.loader import ArrayLoader
from models.paid import PAID
from models.phys_vae import PhysVAE
from utils import load_config


def evaluate(
    model: nnx.Module,
    testloader: ArrayLoader,
    local: bool = False,
    *,
    key: PRNGKeyArray,
):
    """Evaluate a model over a test loader via its __call__(x, key) protocol.

    Both PAID and PhysVAE return (forecast, coefficient) from their call,
    so the evaluation is shared. Models outside the physics pair do not
    follow this protocol and are not supported.

    Returns the forecast MAE and the coefficient MAE.
    """
    assert isinstance(model, (PAID, PhysVAE)), f"{type(model)} is not supported."
    model.eval()

    forecast_mae = None
    coeff_mae = jnp.zeros((0, 1))
    pbar = tqdm(testloader, desc="Evaluating") if local else testloader
    for batch, coeffs in pbar:
        gen_key, key = jax.random.split(key)
        forecast, coeff_pred = model(batch, gen_key)

        if forecast_mae is None:
            forecast_mae = jnp.absolute(forecast - batch)
        else:
            forecast_mae = jnp.concatenate(
                [forecast_mae, jnp.absolute(forecast - batch)]
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
        config["dataset_config"],
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
