import time
from argparse import ArgumentParser
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime

import jax
import optax
import trackio
from flax import nnx
from jax.random import split
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import DATASETS
from models import LOSS_FNS, LOSS_LABELS, MODELS
from utils import load_config


@dataclass
class TrainResult:
    """Result of a training run."""

    model: nnx.Module
    val_loader: DataLoader
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
    local: bool = False,
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
    local : bool = False
        Whether the program should output information
    key : jax.Array
        PRNG key for reproducibility during training

    Returns
    -------
    TrainResult
        Dataclass containing the trained model, validation loader, and training metrics
    """
    # Load datasets
    data_key, key = split(key, num=2)
    train_loader, val_loader, _ = DATASETS[config["dataset"]](
        **config["dataset_config"], key=data_key
    )

    # Build model using provided rngs
    model = MODELS[config["model"]](
        **config["model_config"],
        dataset_config=config["dataset_config"],
        rngs=rngs,
    )

    # Get loss function and loss labels
    loss_fn = LOSS_FNS[config["model"]]
    loss_labels = LOSS_LABELS[config["model"]]

    # Extract training parameters
    epochs = config["trainer_config"]["epochs"]
    learning_rate = config["trainer_config"]["learning_rate"]
    eps = config["trainer_config"].get("eps", 1e-8)
    weight_decay = config["trainer_config"].get("weight_decay", 0.0)
    grad_clip = config["trainer_config"].get("grad_clip", None)

    # Setup optimizer; add_decayed_weights matches torch Adam's L2 weight decay
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

    # Generic train step over the model's loss function; the validation step
    # calls the loss directly so no gradients or optimizer updates occur
    @nnx.jit
    def train_step(
        model: nnx.Module, data: jax.Array, optimizer: nnx.Optimizer, key: jax.Array
    ) -> jax.Array:
        (_, losses), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model, data, key)
        optimizer.update(model, grads)
        return losses

    val_step = nnx.jit(loss_fn)

    # Training loop
    final_loss = 0.0
    best_loss = float("inf")
    best_epoch = 0

    pbar = tqdm(range(epochs)) if local else range(epochs)

    print("\nStarting training steps...\n")

    start = time.perf_counter()
    for epoch in pbar:
        running_loss = defaultdict(float)

        train_pbar = (
            tqdm(train_loader, desc="Train batches", leave=False)
            if local
            else train_loader
        )
        for batch, _ in train_pbar:
            key, train_key = split(key, num=2)
            losses = train_step(
                model=model, data=batch, optimizer=optimizer, key=train_key
            )

            assert len(losses.shape) == 1, "Only vector loss outputs can be processed"

            if not loss_labels:
                loss_labels = ["loss"] + [
                    f"sub_loss_{i + 1}" for i in range(len(losses) - 1)
                ]

            for loss_label, loss in zip(loss_labels, losses):
                running_loss[loss_label] += loss.item() / len(train_loader)

        model.eval()
        val_pbar = (
            tqdm(val_loader, desc="Validation batches", leave=False)
            if local
            else val_loader
        )
        for batch, _ in val_pbar:
            key, val_key = split(key, num=2)
            losses = val_step(model=model, data=batch, key=val_key)[1]

            assert len(losses.shape) == 1, "Only vector loss outputs can be processed"

            running_loss["val_loss"] += losses[0].item() / len(val_loader)

        model.train()

        # Log metrics per epoch (trackio is global singleton, assumed initialized by caller)
        trackio.log(running_loss)

        curr_loss = running_loss["loss"]
        if best_loss >= curr_loss:
            best_loss = curr_loss
            best_epoch = epoch

        if local:
            assert isinstance(pbar, tqdm)
            pbar.set_description(
                f"Training, loss={running_loss['loss']:.2E}, val_loss={running_loss['val_loss']:.2E}"
            )

        final_loss = running_loss["loss"]

    end = time.perf_counter()
    train_time = end - start

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
    parser.add_argument("--local", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    print("\nConfig loaded.")

    key = jax.random.key(args.seed)
    rngs = nnx.Rngs(args.seed)

    # Initialize trackio for standalone execution
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

    result = train(config, rngs=rngs, local=args.local, key=init_key)

    print("\nTraining complete.")
    print(f"Final loss: {result.final_loss:.2E}")
    print(f"Best loss: {result.best_loss:.2E}")
    print(f"Best epoch: {result.best_epoch}")
    print(f"Train time: {format_time(result.train_time)}")

    trackio.finish()
