import os
from argparse import ArgumentParser
from datetime import datetime

import matplotlib.pyplot as plt
from flax import nnx
from jax.random import PRNGKey

from datasets import get_dataset
from models import load_model
from utils.config import ExperimentConfig, load_config
from visualisers.ddim import generate
from visualisers.vae import reconstruct

NOW = datetime.now().strftime("%y%m%d_%H%M%S")


VISUALISER = {"vae": reconstruct, "ddim": generate}


def visualise(
    config: ExperimentConfig,
    folder: os.PathLike,
    frmt: str = "png",
    checkpoint: str = "latest",
    n_rows: int = 5,
    n_cols: int = 20,
) -> None:
    """
    Visualise the generated (or reconstructed) output produced by the specified trained model.

    Parameters
    ----------
    config : ExperimentConfig
        The config of the trained model
    folder : PathLike
        The output folder to store the output visualisation
    frmt : str = png
        The format of the output
    checkpoint : str = latest
        The checkpoint to use
    n_rows : int = 5
        The number of rows of samples
    n_cols : int = 20
        The number of columns of samples
    """
    print("Setting up visualisation...\n")
    rngs = nnx.Rngs(27032002)
    key = PRNGKey(0)

    model = load_model(
        config.model,
        config.model_config,
        os.path.join(os.getcwd(), config.checkpoint_dir, checkpoint),
        rngs=rngs,
    )

    _, test_loader = get_dataset(config.dataset, config.dataset_config)
    images = next(iter(test_loader))[0][: n_rows * n_cols]

    print("Generating samples...\n")
    grid = (
        VISUALISER[config.model](model, images, key=key)
        .reshape(n_rows, n_cols, 28, 28, 1)
        .transpose(0, 2, 1, 3, 4)
        .reshape(n_rows * 28, n_cols * 28, 1)
    )

    print("Creating visualisation...\n")
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    fig.tight_layout()

    ax.set_xticks([])
    ax.set_yticks([])

    ax.imshow(grid, cmap="gray")

    if not os.path.isdir(folder):
        os.makedirs(folder)

    perm = os.path.join(folder, f"{NOW}.{frmt}")
    latest = os.path.join(folder, f"latest.{frmt}")

    plt.savefig(perm, format=frmt)
    plt.savefig(latest, format=frmt)

    print(f"Visualisations saved to {perm} and {latest}.")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", help="path to config file", required=True)
    parser.add_argument(
        "-o", "--output", help="output folder for images", required=True
    )
    parser.add_argument("--format", help="output image format", default="png")
    parser.add_argument(
        "--checkpoint", help="checkpoint file in YYMMDD_HHMMSS format", default="latest"
    )
    args = parser.parse_args()
    config = load_config(args.config)

    visualise(config, args.output, args.format, args.checkpoint)
