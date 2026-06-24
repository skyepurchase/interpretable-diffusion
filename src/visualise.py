import os
from argparse import ArgumentParser
from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns
from flax import nnx
from jax.random import PRNGKey

from datasets import get_dataset
from models import load_model
from utils.config import ExperimentConfig, VisualiserConfig, load_config
from utils.config.datasets import get_full_dataset_name
from visualisers.ddim import generate_mnist, generate_timeseries
from visualisers.vae import (
    reconstruct_1D_series,
    reconstruct_images,
    reconstruct_point_series,
)

NOW = datetime.now().strftime("%y%m%d_%H%M%S")


VISUALISER = {
    "mnist": {
        "vae": reconstruct_images,
        "ddim": generate_mnist,
    },
    "pivae-pendulum": {"phys_vae": reconstruct_point_series},
    "pivae-diffusion": {
        "phys_vae": reconstruct_1D_series,
        "paid": generate_timeseries,
    },
    "pdebench-burgers": {
        "paid": generate_timeseries,
    },
    "burgers": {
        "paid": generate_timeseries,
    },
}


sns.set_theme(
    font="DejaVu Sans",
    rc={
        "axes.axisbelow": False,
        "axes.edgecolor": "lightgrey",
        "axes.facecolor": "None",
        "axes.grid": False,
        "axes.labelcolor": "dimgrey",
        "axes.spines.right": False,
        "axes.spines.top": False,
        "figure.facecolor": "white",
        "lines.solid_capstyle": "round",
        "patch.edgecolor": "w",
        "patch.force_edgecolor": True,
        "text.color": "dimgrey",
        "xtick.bottom": False,
        "xtick.color": "dimgrey",
        "xtick.direction": "out",
        "xtick.top": False,
        "ytick.color": "dimgrey",
        "ytick.direction": "out",
        "ytick.left": False,
        "ytick.right": False,
    },
)
sns.set_palette(sns.color_palette("muted"))


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
    dataset_name = get_full_dataset_name(exp_cfg.dataset, exp_cfg.dataset_config)
    raw = VISUALISER[dataset_name][exp_cfg.model](model, inputs, key=key)
    shape = raw.shape[1:]

    print("Creating visualisation...\n")
    if len(shape) == 3:
        print("Assuming image data.\n")

        grid = (
            raw.reshape(vis_cfg.n_rows, vis_cfg.n_cols, *shape)
            .transpose(0, 2, 1, 3, 4)
            .reshape(vis_cfg.n_rows * shape[0], vis_cfg.n_cols * shape[1], shape[2])
        )

        fig, ax = plt.subplots(1, 1, figsize=(12, 5))
        fig.tight_layout()

        ax.set_xticks([])
        ax.set_yticks([])

        colour = "gray" if dataset_name == "mnist" else None
        ax.imshow(grid, cmap=colour)

        plt.savefig(perm, format=vis_cfg.format)
        plt.savefig(latest, format=vis_cfg.format)
    elif len(shape) == 1:
        print("Assuming point data.\n")

        fig, axes = plt.subplots(
            vis_cfg.n_rows, vis_cfg.n_cols, figsize=(12, 5), sharex=True, sharey=True
        )
        fig.tight_layout()

        fig.suptitle("Sample of force-damped pendulum series")

        i = 0
        for j, row in enumerate(axes):
            for ax in row:
                ax.plot(timesteps[i], raw[i], label="prediction")
                ax.plot(timesteps[i], inputs[i], label="ground truth")
                if j == len(axes) - 1:
                    ax.set_xlabel("timestep")
                    ax.set_xticks([0, 1, 2, 2.5], ["0", "1", "2", "2.5"])

                i += 1

            row[0].set_ylim(bottom=-2, top=2)
            row[0].set_ylabel("pendulum angle")

        axes[-1][-1].legend()

        plt.savefig(perm, format=vis_cfg.format)
        plt.savefig(latest, format=vis_cfg.format)
    else:
        raise NotImplementedError(f"Data shape {shape} not supported")

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
