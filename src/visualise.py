import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from flax import nnx
from jax._src.array import PRNGKeyArray

from datasets import DATASETS
from datasets.loader import ArrayLoader
from models import load_model
from models.paid import PAID
from utils import load_config

sns.set_theme(
    font="DejaVu Sans",
    font_scale=1.75,
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
    out = jnp.permute_dims(out, (0, 2, 1))
    target = jnp.permute_dims(target, (0, 2, 1))
    error = jnp.abs(out - target)

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

    fig, axes = plt.subplots(num, 3, figsize=(15, 2.0 * num), sharex=True, sharey=True)
    fig.subplots_adjust(
        left=0.07, right=0.88, bottom=0.15, top=0.94, wspace=0.20, hspace=0.32
    )
    axes = np.atleast_2d(axes)

    for i, row in enumerate(axes):
        vmin = float(jnp.minimum(out[i].min(), target[i].min()))
        vmax = float(jnp.maximum(out[i].max(), target[i].max()))

        img = row[0].imshow(
            out[i],
            vmin=vmin,
            vmax=vmax,
            cmap="viridis",
            origin="lower",
            extent=extent,
            aspect="auto",
        )
        row[1].imshow(
            target[i],
            vmin=vmin,
            vmax=vmax,
            cmap="viridis",
            origin="lower",
            extent=extent,
            aspect="auto",
        )

        pos = row[2].get_position()
        row[2].set_position([pos.x0 + 0.025, pos.y0, pos.width, pos.height])
        err = row[2].imshow(
            error[i],
            cmap="magma",
            origin="lower",
            extent=extent,
            aspect="auto",
        )

        fig.colorbar(img, cax=row[1].inset_axes([1.025, 0, 0.025, 1]))
        fig.colorbar(err, cax=row[2].inset_axes([1.025, 0, 0.025, 1]))

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
    axes[0, 0].legend(
        loc="upper right", labelcolor="white", markerscale=5, frameon=False
    )

    for ax in axes[-1]:
        ax.set_xlabel("time (s)")
    for row in axes:
        row[0].set_ylabel("x")

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
    model: PAID,
    test_loader: ArrayLoader,
    folder: os.PathLike | str,
    length: float,
    stamp: str,
    format: str = "png",
    num: int = 5,
    *,
    key: PRNGKeyArray,
) -> None:
    """
    Visualise the output produced by a trained PAID model.

    Parameters
    ----------
    model : PAID
        The trained model to generate visualisations
    test_loader : ArrayLoader
        The test set loader
    folder : os.PathLike | str
        The output folder for the images
    length : float
        The length of the dimension
    stamp : str
        The timestamp for this run
    format : str
        The output image format
    num : int
        The number of outputs to visualise
    key : PRNGKeyArray
        The random key
    """
    print("Setting up visualisation...\n")
    gen_key, sample_key = jax.random.split(key, num=2)

    if not os.path.isdir(folder):
        os.makedirs(folder)

    perm = os.path.join(folder, f"{model.condition}_{stamp}.{format}")
    latest = os.path.join(folder, f"latest.{format}")

    assert isinstance(model, PAID), (
        f"Visualisation supports PAID models but received {type(model).__name__}."
    )

    inputs = test_loader.data

    if inputs.ndim == 4:
        inputs = jnp.squeeze(inputs, axis=-1)

    inputs = inputs[:, : model.window]

    print("Generating samples...\n")
    out, _ = model(inputs, gen_key)

    err = jnp.absolute(out[:, model.condition :] - inputs[:, model.condition :]).mean(
        (1, 2)
    )
    err_mean = err.mean()
    err_std = err.std()
    print(err_mean, err_std, jnp.min(err))

    best_sample = jnp.arange(inputs.shape[0])[err < err_mean - err_std]
    print(best_sample.shape)
    ids = jax.random.permutation(sample_key, best_sample)[:num]

    inputs = inputs[ids]
    out = out[ids]

    times = jnp.arange(model.window) * model.dt
    condition_time = model.condition * model.dt

    print("Creating visualisation...\n")
    if model.num_grid_points > 1:
        fig = visualise_2d(out, inputs, times, condition_time, length, num)
    else:
        fig = visualise_1d(out, inputs, times, condition_time, num)

    plt.savefig(perm, format=format)
    plt.savefig(latest, format=format)
    plt.close(fig)

    print(f"Visualisations saved to {perm} and {latest}.")


if __name__ == "__main__":
    from argparse import ArgumentParser
    from datetime import datetime

    NOW = datetime.now().strftime("%y%m%d_%H%M%S")

    parser = ArgumentParser()
    parser.add_argument("-c", "--config", help="path to config file", required=True)
    parser.add_argument(
        "-o", "--output", help="output folder for images", required=True
    )
    parser.add_argument("--format", help="output image format", default="png")
    parser.add_argument(
        "--checkpoint",
        help="the observation period the model was trained on",
        default="latest",
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
    config = load_config(args.config)

    rngs = nnx.Rngs(args.seed)
    key = jax.random.key(args.seed)
    vis_key, data_key = jax.random.split(key, num=2)

    model = load_model(
        config["model"],
        config["model_config"],
        config["dataset_config"],
        os.path.join(os.getcwd(), config["checkpoint_dir"], args.checkpoint),
        rngs=rngs,
    )

    _, _, test_loader = DATASETS[config["dataset"]](
        **config["dataset_config"], key=data_key
    )

    visualise(
        model,
        test_loader,
        folder=args.output,
        length=config["dataset_config"]["length"],
        stamp=NOW,
        format=args.format,
        num=args.num,
        key=vis_key,
    )
