import os
import shutil
import time
from argparse import ArgumentParser
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from functools import partial

import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from flax import nnx
from jax.random import split

from datasets import DATASETS
from datasets.loader import ArrayLoader
from models import LOSS_FNS, LOSS_LABELS, MODELS
from models.paid import PAID
from utils import load_config


@dataclass
class TrainResult:
    """Result of a training run."""

    model: nnx.Module
    val_loader: ArrayLoader
    final_loss: float
    best_loss: float
    best_epoch: int
    train_time: float


def format_time(seconds: float) -> str:
    """Convert seconds into reasonable precision"""
    if seconds < 180:
        return f"{seconds:.2f} seconds"
    elif seconds < 7200:
        return f"{seconds / 60:.2f} minutes"
    elif seconds < 86400:
        return f"{seconds / 3600:.2f} hours"
    else:
        return f"{seconds / 86400:.2f} days?!?!"


def train(
    config: dict,
    rngs: nnx.Rngs,
    progress: bool = False,
    *,
    key: jax.Array,
) -> TrainResult:
    """
    Train a model given a config.

    Parameters
    ----------
    config : dict
        Full configuration dictionary with dataset, model, model_config, trainer_config
    rngs : nnx.Rngs
        Flax random number generators for model initialization
    progress : bool = False
        Whether to show progress bars and log metrics to trackio
    key : jax.Array
        PRNG key for reproducibility during training

    Returns
    -------
    TrainResult
        Dataclass containing the trained model, validation loader, and training metrics
    """
    data_key, key = split(key, num=2)
    train_loader, val_loader, _ = DATASETS[config["dataset"]](
        **config["dataset_config"], key=data_key
    )

    model = MODELS[config["model"]](
        **config["model_config"],
        dataset_config=config["dataset_config"],
        rngs=rngs,
    )

    window = model.window if isinstance(model, PAID) else None

    loss_fn = LOSS_FNS[config["model"]]
    loss_labels = LOSS_LABELS[config["model"]]

    epochs = config["trainer_config"]["epochs"]
    learning_rate = config["trainer_config"]["learning_rate"]
    eps = config["trainer_config"].get("eps", 1e-8)
    weight_decay = config["trainer_config"].get("weight_decay", 0.0)
    grad_clip = config["trainer_config"].get("grad_clip", None)

    gradient_transform = optax.chain(
        optax.add_decayed_weights(weight_decay),
        optax.adam(learning_rate, eps=eps),
    )
    if grad_clip is not None:
        gradient_transform = optax.chain(
            optax.clip(grad_clip),
            gradient_transform,
        )
    optimizer = nnx.Optimizer(model, gradient_transform, wrt=nnx.Param)

    @nnx.jit
    def train_step(
        model: nnx.Module, data: jax.Array, optimizer: nnx.Optimizer, key: jax.Array
    ) -> jax.Array:
        (_, losses), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model, data, key)
        optimizer.update(model, grads)
        return losses

    if config["model"] == "paid":
        # PAID skips aug_loss ODE solve
        val_step = nnx.jit(partial(loss_fn, compute_aug=False))
    else:
        val_step = nnx.jit(loss_fn)

    val_rec_idx = 1 if config["model"] in ("phys_vae", "paid") else 0

    val_ckpt_dir = os.path.join(config["checkpoint_dir"], "val-checkpoint")
    os.makedirs(val_ckpt_dir, exist_ok=True)
    checkpointer = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
    ckpt_interval = config["trainer_config"].get("ckpt_interval", 5)
    saved_epochs: list[int] = []

    final_loss = 0.0
    val_history: list[jax.Array] = []

    if progress:
        from tqdm import tqdm

        pbar = tqdm(range(epochs))
    else:
        pbar = range(epochs)

    print("\nStarting training steps...\n")

    start = time.perf_counter()
    for epoch in pbar:
        train_running = None

        train_pbar = (
            tqdm(train_loader, desc="Train batches", leave=False)
            if progress
            else train_loader
        )
        for batch, _ in train_pbar:
            if window is not None:
                batch = batch[:, :window]
            key, train_key = split(key, num=2)
            losses = train_step(
                model=model, data=batch, optimizer=optimizer, key=train_key
            )

            assert len(losses.shape) == 1, "Only vector loss outputs can be processed"

            if not loss_labels:
                loss_labels = ["loss"] + [
                    f"sub_loss_{i + 1}" for i in range(len(losses) - 1)
                ]

            train_running = losses if train_running is None else train_running + losses

        train_mean = train_running / len(train_loader)

        model.eval()
        val_running = None
        val_pbar = (
            tqdm(val_loader, desc="Validation batches", leave=False)
            if progress
            else val_loader
        )
        for batch, _ in val_pbar:
            if window is not None:
                batch = batch[:, :window]
            key, val_key = split(key, num=2)
            losses = val_step(model=model, data=batch, key=val_key)[1]

            assert len(losses.shape) == 1, "Only vector loss outputs can be processed"

            vals = jnp.stack([losses[0], losses[val_rec_idx]])
            val_running = vals if val_running is None else val_running + vals

        model.train()

        val_mean = val_running / len(val_loader)

        val_history.append(val_mean[1])

        if progress:
            import trackio

            running_loss = defaultdict(float)
            for loss_label, value in zip(loss_labels, train_mean.tolist()):
                running_loss[loss_label] = value
            running_loss["val_loss"] = float(val_mean[0])
            running_loss["val_reconstruction"] = float(val_mean[1])
            trackio.log(running_loss)

        if (epoch + 1) % ckpt_interval == 0 or epoch == epochs - 1:
            _, state = nnx.split(model)
            checkpointer.save(
                os.path.join(os.getcwd(), val_ckpt_dir, str(epoch)), state
            )
            saved_epochs.append(epoch)

        if progress:
            pbar.set_description(
                f"Training, loss={running_loss['loss']:.2E}, val_loss={running_loss['val_loss']:.2E}"
            )

        final_loss = train_mean[0]

    end = time.perf_counter()
    train_time = end - start
    final_loss = float(final_loss)

    # Select best model from regular checkpoints
    checkpointer.wait_until_finished()
    history = jnp.stack(val_history)
    saved_mask = jnp.array(
        [epoch in set(saved_epochs) for epoch in range(epochs)], dtype=bool
    )
    best_epoch = int(jnp.argmin(jnp.where(saved_mask, history, jnp.inf)))
    best_loss = float(history[best_epoch])

    graph, state = nnx.split(model)
    restore_checkpointer = ocp.StandardCheckpointer()
    best_state = restore_checkpointer.restore(
        os.path.join(os.getcwd(), val_ckpt_dir, str(best_epoch)), state
    )
    model = nnx.merge(graph, best_state)

    for epoch in saved_epochs:
        shutil.rmtree(os.path.join(os.getcwd(), val_ckpt_dir, str(epoch)))

    return TrainResult(
        model=model,
        val_loader=val_loader,
        final_loss=final_loss,
        best_loss=best_loss,
        best_epoch=best_epoch,
        train_time=train_time,
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", help="path to config file", required=True)
    parser.add_argument("-s", "--seed", type=int, help="seed for the run", default=0)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    print("\nConfig loaded.")

    key = jax.random.key(args.seed)
    rngs = nnx.Rngs(args.seed)

    stamp = datetime.now().strftime("%m%d-%H%M")
    run_name = "_".join(
        [
            config["model"],
            config["dataset"],
            config["dataset_config"].get("system", ""),
            stamp,
        ]
    )
    run_name = run_name.replace("__", "_")

    # Load dataset to get batch size for trackio config
    data_key, init_key = jax.random.split(key)
    train_loader_temp, val_loader_temp, _ = DATASETS[config["dataset"]](
        **config["dataset_config"], key=data_key
    )
    batch_size = next(iter(train_loader_temp))[0].shape[0]

    if args.progress:
        try:
            import trackio
        except ImportError as err:
            raise SystemExit(
                "trackio is not installed; install the dashboard extra with "
                "`uv sync --extra dashboard`"
            ) from err

        trackio.init(
            project="Interpretable Diffusion",
            name=run_name,
            config={
                "epochs": config["trainer_config"]["epochs"],
                "learning_rate": config["trainer_config"]["learning_rate"],
                "batch_size": batch_size,
            },
        )

    print("\nSetup complete.")

    result = train(config, rngs=rngs, progress=args.progress, key=init_key)

    print("\nTraining complete.")
    print(f"Final loss: {result.final_loss:.2E}")
    print(f"Best loss: {result.best_loss:.2E}")
    print(f"Best epoch: {result.best_epoch}")
    print(f"Train time: {format_time(result.train_time)}")

    if args.progress:
        trackio.finish()
