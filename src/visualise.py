import os
from argparse import ArgumentParser
from datetime import datetime

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from flax import nnx

from datasets import DATASETS
from models import load_model
from models.paid import PAID
from utils import load_config

NOW = datetime.now().strftime("%y%m%d_%H%M%S")


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


def visualise_2d(
    out: jax.Array,
    target: jax.Array,
    times: jax.Array,
    condition_time: float,
    length: float,
    num: int,
) -> plt.Figure:
    """
    Visualise 2D samples as a grid of PAID output, target, and error images.

    Each row shows one sample: the PAID output and the target share a
    colour scale and colourbar, the absolute error has its own. A vline
    marks the condition/generated boundary.

    Parameters
    ----------
    out : Array (B, T, X)
        The generated samples
    target : Array (B, T, X)
        The ground truth samples
    times : Array (window)
        The time of each frame in seconds
    condition_time : float
        The time of the condition/generated boundary in seconds
    length : float
        The length of the spatial domain
    num : int
        The number of samples visualised

    Returns
    -------
    fig : Figure
        The created figure
    """
    # (B, T, X) -> (B, X, T) so time runs horizontally
    out = jnp.permute_dims(out, (0, 2, 1))
    target = jnp.permute_dims(target, (0, 2, 1))
    error = jnp.abs(out - target)

    # Output and target share one colour scale
    vmin = float(jnp.minimum(out.min(), target.min()))
    vmax = float(jnp.maximum(out.max(), target.max()))

    # Align cell centres with the sample points: frame i at times[i],
    # grid point j at j * length / (X - 1)
    frame_dt = float(times[1] - times[0])
    x_grid = out.shape[1]
    ds = length / (x_grid - 1) if length > 0 else 1.0
    y_min = -0.5 * ds if length > 0 else -0.5
    y_max = length + 0.5 * ds if length > 0 else x_grid - 0.5
    extent = (
        float(times[0]) - 0.5 * frame_dt,
        float(times[-1]) + 0.5 * frame_dt,
        y_min,
        y_max,
    )

    fig, axes = plt.subplots(num, 3, figsize=(12, 2.5 * num), sharex=True, sharey=True)
    fig.tight_layout()
    axes = np.atleast_2d(axes)

    images = []
    for i, row in enumerate(axes):
        images.append(
            [
                row[0].imshow(
                    out[i],
                    vmin=vmin,
                    vmax=vmax,
                    cmap="viridis",
                    origin="lower",
                    extent=extent,
                    aspect="auto",
                ),
                row[1].imshow(
                    target[i],
                    vmin=vmin,
                    vmax=vmax,
                    cmap="viridis",
                    origin="lower",
                    extent=extent,
                    aspect="auto",
                ),
                row[2].imshow(
                    error[i],
                    cmap="magma",
                    origin="lower",
                    extent=extent,
                    aspect="auto",
                ),
            ]
        )
        for ax in row:
            ax.axvline(
                condition_time,
                color="red",
                linestyle="--",
                linewidth=1,
                label="condition" if i == 0 else None,
            )

    axes[0, 0].set_title("PAID output")
    axes[0, 1].set_title("Target")
    axes[0, 2].set_title("Absolute error")
    axes[0, 0].legend(loc="upper right")

    for ax in axes[-1]:
        ax.set_xlabel("time (s)")
    for row in axes:
        row[0].set_ylabel("x")

    # Colorbars to the right of the target and error columns
    fig.colorbar(images[-1][1], ax=list(axes[:, 1]), location="right")
    fig.colorbar(images[-1][2], ax=list(axes[:, 2]), location="right")

    return fig


def visualise_1d(
    out: jax.Array,
    target: jax.Array,
    times: jax.Array,
    condition_time: float,
    num: int,
) -> plt.Figure:
    """
    Visualise 1D samples as overlaid output/target lineplots.

    Each row shows one sample with the PAID output and the target
    overlaid. A vline marks the condition/generated boundary.

    Parameters
    ----------
    out : Array (B, T) or (B, T, 1)
        The generated samples
    target : Array (B, T) or (B, T, 1)
        The ground truth samples
    times : Array (window)
        The time of each frame in seconds
    condition_time : float
        The time of the condition/generated boundary in seconds
    num : int
        The number of samples visualised

    Returns
    -------
    fig : Figure
        The created figure
    """
    if out.ndim == 3:
        out = jnp.squeeze(out, axis=-1)
    if target.ndim == 3:
        target = jnp.squeeze(target, axis=-1)

    fig, axes = plt.subplots(num, 1, figsize=(12, 2 * num), sharex=True, sharey=True)
    fig.tight_layout()
    axes = np.atleast_1d(axes)

    for i, ax in enumerate(axes):
        ax.plot(times, target[i], label="target")
        ax.plot(times, out[i], label="PAID output")
        ax.axvline(
            condition_time,
            color="red",
            linestyle="--",
            linewidth=1,
            label="condition" if i == 0 else None,
        )
        ax.set_ylabel("signal")

    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("time (s)")

    return fig


def visualise(
    model: str,
    model_config: dict,
    dataset: str,
    dataset_config: dict,
    checkpoint_dir: str,
    folder: os.PathLike | str,
    format: str = "png",
    checkpoint: str = "latest",
    num: int = 5,
    seed: int = 0,
    **_,
) -> None:
    """
    Visualise the output produced by a trained PAID model.

    Parameters
    ----------
    model : str
        The name of the model, must be "paid"
    model_config : dict
        The config of the trained model
    dataset : str
        The name of the dataset
    dataset_config : dict
        The config of the dataset
    checkpoint_dir : str
        The directory the checkpoint is stored in
    folder : os.PathLike | str
        The output folder for the images
    format : str
        The output image format
    checkpoint : str
        The checkpoint file to load
    num : int
        The number of outputs to visualise
    seed : int
        The seed for the run
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
    assert isinstance(loaded_model, PAID), (
        f"Visualisation supports PAID models but received {type(loaded_model).__name__}."
    )

    data_key = jax.random.key(0)
    _, _, test_loader = DATASETS[dataset](**dataset_config, key=data_key)
    inputs = next(iter(test_loader))[0][:num]

    # Drop any trailing channel axis; PAID expects (B, T, X)
    if inputs.ndim == 4:
        inputs = jnp.squeeze(inputs, axis=-1)

    # Slice inputs to PAID's window
    inputs = inputs[:, : loaded_model.window]

    print("Generating samples...\n")
    out, _ = loaded_model(inputs, key)

    # Real time of each frame and of the condition/generated boundary
    frame_dt = dataset_config["time"] / dataset_config["t_grid"]
    times = jnp.arange(loaded_model.window) * frame_dt
    condition_time = loaded_model.condition * frame_dt

    print("Creating visualisation...\n")
    if dataset_config["x_grid"] > 1:
        fig = visualise_2d(
            out, inputs, times, condition_time, dataset_config["length"], num
        )
    else:
        fig = visualise_1d(out, inputs, times, condition_time, num)

    plt.savefig(perm, format=format)
    plt.savefig(latest, format=format)
    plt.close(fig)

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
    parser.add_argument(
        "--num",
        type=int,
        help="the number of outputs to visualise",
        default=5,
    )

    args = parser.parse_args()
    exp_cfg = load_config(args.config)

    visualise(
        **exp_cfg,
        folder=args.output,
        format=args.format,
        checkpoint=args.checkpoint,
        num=args.num,
        seed=args.seed,
    )
