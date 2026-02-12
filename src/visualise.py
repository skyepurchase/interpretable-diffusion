import os
from argparse import ArgumentParser
from datetime import datetime

import matplotlib.pyplot as plt
from flax import nnx
from jax.random import PRNGKey

from datasets import get_dataset
from models import load_model
from utils.config import ExperimentConfig, VisualiserConfig, load_config
from visualisers.ddim import generate
from visualisers.vae import reconstruct

NOW = datetime.now().strftime("%y%m%d_%H%M%S")


VISUALISER = {"vae": reconstruct, "ddim": generate}


def visualise(
    exp_cfg: ExperimentConfig,
    vis_cfg: VisualiserConfig,
) -> None:
    """
    Visualise the generated (or reconstructed) output produced by the specified trained model.

    Parameters
    ----------
    exp_cfg : ExperimentConfig
        The config of the trained model
    vis_cfg : VisualiserConfig
        The config for the visualisation
    """
    print("Setting up visualisation...\n")
    rngs = nnx.Rngs(vis_cfg.seed)
    key = PRNGKey(vis_cfg.seed)

    if not os.path.isdir(vis_cfg.folder):
        os.makedirs(vis_cfg.folder)

    perm = os.path.join(vis_cfg.folder, f"{NOW}.{vis_cfg.format}")
    latest = os.path.join(vis_cfg.folder, f"latest.{vis_cfg.format}")

    model = load_model(
        exp_cfg.model,
        exp_cfg.model_config,
        os.path.join(os.getcwd(), exp_cfg.checkpoint_dir, vis_cfg.checkpoint),
        rngs=rngs,
    )

    _, _, test_loader = get_dataset(exp_cfg.dataset, exp_cfg.dataset_config)
    inputs = next(iter(test_loader))[0][: vis_cfg.n_rows * vis_cfg.n_cols]
    timesteps = next(iter(test_loader))[1][: vis_cfg.n_rows * vis_cfg.n_cols]

    print("Generating samples...\n")
    raw = VISUALISER[exp_cfg.model](model, inputs, key=key)

    print("Creating visualisation...\n")
    if len(raw.shape) == 4:  # Very ugly
        # Assume output was images
        grid = (
            raw.reshape(vis_cfg.n_rows, vis_cfg.n_cols, 28, 28, 1)
            .transpose(0, 2, 1, 3, 4)
            .reshape(vis_cfg.n_rows * 28, vis_cfg.n_cols * 28, 1)
        )

        fig, ax = plt.subplots(1, 1, figsize=(12, 5))
        fig.tight_layout()

        ax.set_xticks([])
        ax.set_yticks([])

        ax.imshow(grid, cmap="gray")

        plt.savefig(perm, format=vis_cfg.format)
        plt.savefig(latest, format=vis_cfg.format)
    else:
        # Assume output was time series
        fig, axes = plt.subplots(
            vis_cfg.n_rows, vis_cfg.n_cols, figsize=(12, 5), sharex=True, sharey=True
        )
        fig.tight_layout()

        i = 0
        for row in axes:
            for ax in row:
                ax.plot(timesteps[i], raw[i])
                ax.plot(timesteps[i], inputs[i])
                i += 1

        plt.savefig(perm, format=vis_cfg.format)
        plt.savefig(latest, format=vis_cfg.format)

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
    parser.add_argument(
        "-s", "--seed", type=int, help="the seed for the run", default=0
    )
    parser.add_argument("--n_rows", type=int, help="the number of rows", default=5)
    parser.add_argument("--n_cols", type=int, help="the number of rows", default=20)

    args = parser.parse_args()
    exp_cfg = load_config(args.config)

    visualise(
        exp_cfg,
        VisualiserConfig(
            folder=args.output,
            format=args.format,
            checkpoint=args.checkpoint,
            n_rows=args.n_rows,
            n_cols=args.n_cols,
            seed=args.seed,
        ),
    )
