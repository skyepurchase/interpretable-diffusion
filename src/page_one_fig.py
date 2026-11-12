import copy
import os
from argparse import ArgumentParser

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
from flax import nnx

from datasets import DATASETS
from models import load_model
from models.paid import PAID
from utils import load_config

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


def generate(
    config: dict,
    conditions: list[int],
    checkpoints: list[str],
    sample: int = 0,
    all_samples: bool = False,
    seed: int = 0,
) -> tuple[list[tuple[int, int, jax.Array]], jax.Array]:
    """
    Generate the outputs of one model per observation period.

    One PAID model is loaded per condition/checkpoint pair and run over the
    same test-set sample(s); only the generated frames of each output are
    kept, the condition frames are carried through sampling unchanged.

    Parameters
    ----------
    config : dict
        The experiment config
    conditions : list[int]
        The observation period of each model
    checkpoints : list[str]
        The checkpoint of each model, one per condition
    sample : int
        The test-set sample to generate
    all_samples : bool
        Generate every test-set sample instead of just ``sample``
    seed : int
        The seed for the run

    Returns
    -------
    outputs : list[(condition, window, out)]
        One entry per model in ascending order of condition. ``out`` is an
        Array (S, window - condition) holding the generated frames of each
        requested sample.
    test_data : Array
        The full unshuffled test set
    indices : Array (S)
        The test-set indices the outputs correspond to
    """
    rngs = nnx.Rngs(seed)
    key = jax.random.key(seed)

    data_key = jax.random.key(0)
    _, _, test_loader = DATASETS[config["dataset"]](
        **config["dataset_config"], key=data_key
    )
    test_data = test_loader.data

    if all_samples:
        indices = jnp.arange(test_data.shape[0])
    else:
        indices = jnp.array([sample])

    batch_size = config["dataset_config"]["batch_size"]

    outputs = []
    for condition, checkpoint in sorted(zip(conditions, checkpoints)):
        model_config = copy.deepcopy(config["model_config"])
        model_config["condition"] = condition

        loaded_model = load_model(
            config["model"],
            model_config,
            config["dataset_config"],
            os.path.join(os.getcwd(), config["checkpoint_dir"], checkpoint),
            rngs=rngs,
        )
        assert isinstance(loaded_model, PAID), (
            f"Visualisation supports PAID models but received {type(loaded_model).__name__}."
        )
        loaded_model.eval()

        window = loaded_model.window
        outs = []
        for start in range(0, len(indices), batch_size):
            batch = test_data[indices[start : start + batch_size]][:, :window]
            out, _ = loaded_model(batch, key)
            if out.ndim == 3:
                out = jnp.squeeze(out, axis=-1)
            outs.append(out[:, condition:])
        outputs.append((condition, window, jnp.concatenate(outs, axis=0)))

        print(f"Condition {condition}: window {window}, output {outputs[-1][2].shape}")

    return outputs, test_data, indices


def page_one_figure(
    truth: jax.Array,
    times: jax.Array,
    outputs: list[tuple[int, int, jax.Array]],
) -> plt.Figure:
    """
    Render the merged observation-period figure for one test sample.

    The ground truth is drawn in SandyBrown up to the largest condition
    (the observations) and in black beyond it (the truth). Each model's
    generated frames are overlaid as a pastel line starting at its own
    condition, in ascending order of condition.

    Parameters
    ----------
    truth : Array (W)
        The ground-truth frames of the sample
    times : Array (W)
        The time of each frame in seconds
    outputs : list[(condition, window, out)]
        One entry per model in ascending order of condition. ``out`` is an
        Array (window - condition) holding the generated frames

    Returns
    -------
    fig : Figure
        The created figure
    """
    k_max = outputs[-1][0]

    fig, ax = plt.subplots(figsize=(12, 3))
    fig.tight_layout()

    ax.plot(times[: k_max + 1], truth[: k_max + 1], color="sandybrown")
    ax.plot(times[k_max:], truth[k_max:], color="black")

    palette = sns.color_palette("pastel", len(outputs))
    for (condition, window, out), colour in zip(outputs, palette):
        ax.plot(times[condition:window], out, color=colour)

    ax.set_xlabel("time (s)")
    ax.set_ylabel("angle")

    return fig


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", help="path to config file", required=True)
    parser.add_argument(
        "--conditions",
        type=int,
        nargs="+",
        help="the observation period of each model",
        required=True,
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        help="the checkpoint of each model, one per condition",
        required=True,
    )
    parser.add_argument(
        "-o", "--output", help="output folder for images", required=True
    )
    parser.add_argument(
        "--sample", type=int, help="the test-set sample to visualise", default=0
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="visualise every test-set sample instead of just --sample",
    )
    parser.add_argument("--format", help="output image format", default="png")
    parser.add_argument(
        "-s", "--seed", type=int, help="the seed for the run", default=0
    )

    args = parser.parse_args()
    assert len(args.conditions) == len(args.checkpoints), (
        f"Received {len(args.conditions)} conditions but {len(args.checkpoints)} checkpoints, they must match."
    )
    exp_cfg = load_config(args.config)

    print("Generating outputs...\n")
    outputs, test_data, indices = generate(
        exp_cfg,
        args.conditions,
        args.checkpoints,
        sample=args.sample,
        all_samples=args.all,
        seed=args.seed,
    )

    print("Creating visualisation...\n")
    frame_dt = exp_cfg["dataset_config"]["time"] / exp_cfg["dataset_config"]["t_grid"]
    max_window = max(window for _, window, _ in outputs)
    times = jnp.arange(max_window) * frame_dt

    if not os.path.isdir(args.output):
        os.makedirs(args.output)

    for i, idx in enumerate(indices):
        truth = test_data[idx, :max_window]
        if truth.ndim > 1:
            truth = jnp.squeeze(truth, axis=-1)

        fig = page_one_figure(truth, times, [(c, w, out[i]) for c, w, out in outputs])

        perm = os.path.join(args.output, f"page_one_s{int(idx):03d}.{args.format}")
        plt.savefig(perm, format=args.format)
        if i == len(indices) - 1:
            latest = os.path.join(args.output, f"latest.{args.format}")
            plt.savefig(latest, format=args.format)
        plt.close(fig)

    print(f"Visualisations saved to {args.output}.")
