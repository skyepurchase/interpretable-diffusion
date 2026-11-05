import os
from argparse import ArgumentParser
from datetime import datetime

import jax
import matplotlib.pyplot as plt
import seaborn as sns
from flax import nnx

from datasets import DATASETS
from models import load_model
from utils import load_config
from visualisers.ddim import generate_timeseries
from visualisers.vae import (
    reconstruct_1D_series,
    reconstruct_point_series,
)

NOW = datetime.now().strftime("%y%m%d_%H%M%S")


VISUALISER = {
    "pivae-pendulum": {"phys_vae": reconstruct_point_series},
    "pivae-diffusion": {
        "phys_vae": reconstruct_1D_series,
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
    model: str,
    model_config: dict,
    dataset: str,
    dataset_config: dict,
    checkpoint_dir: str,
    folder: os.PathLike | str,
    format: str = "png",
    checkpoint: str = "latest",
    n_rows: int = 5,
    n_cols: int = 20,
    seed: int = 0,
    **_,
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
    rngs = nnx.Rngs(seed)
    key = jax.random.key(seed)

    if not os.path.isdir(folder):
        os.makedirs(folder)

    perm = os.path.join(folder, f"{NOW}.{format}")
    latest = os.path.join(folder, f"latest.{format}")

    loaded_model = load_model(
        model,
        model_config,
        dataset_config,
        os.path.join(os.getcwd(), checkpoint_dir, checkpoint),
        rngs=rngs,
    )

    data_key = jax.random.key(0)
    _, _, test_loader = DATASETS[dataset](**dataset_config, key=data_key)
    inputs = next(iter(test_loader))[0][: n_rows * n_cols]
    timesteps = next(iter(test_loader))[1][: n_rows * n_cols]

    # Slice inputs to PAID's window; other models take the full sample
    window = getattr(loaded_model, "window", None)
    if window is not None:
        inputs = inputs[:, :window]

    print("Generating samples...\n")
    if "system" in dataset_config and dataset_config["system"] != "":
        dataset_name = "-".join([dataset, dataset_config["system"]])
    else:
        dataset_name = dataset

    raw = VISUALISER[dataset_name][model](loaded_model, inputs, key=key)
    shape = raw.shape[1:]

    print("Creating visualisation...\n")
    if len(shape) == 3:
        print("Assuming image data.\n")

        grid = (
            raw.reshape(n_rows, n_cols, *shape)
            .transpose(0, 2, 1, 3, 4)
            .reshape(n_rows * shape[0], n_cols * shape[1], shape[2])
        )

        fig, ax = plt.subplots(1, 1, figsize=(12, 5))
        fig.tight_layout()

        ax.set_xticks([])
        ax.set_yticks([])

        ax.imshow(grid)

        plt.savefig(perm, format=format)
        plt.savefig(latest, format=format)
    elif len(shape) == 1:
        print("Assuming point data.\n")

        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(12, 5), sharex=True, sharey=True
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

        plt.savefig(perm, format=format)
        plt.savefig(latest, format=format)
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
        **exp_cfg,
        folder=args.output,
        format=args.format,
        checkpoint=args.checkpoint,
        n_rows=args.n_rows,
        n_cols=args.n_cols,
        seed=args.seed,
    )
