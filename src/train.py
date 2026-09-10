import time
from argparse import ArgumentParser
from collections import defaultdict
from datetime import datetime
from typing import Callable, Optional

import jax
import optax
import trackio
import yaml
from flax import nnx
from jax.random import PRNGKey, split
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import DATASETS
from eval import evaluate
from models import LOSS_LABELS, MODELS, PRETRAIN_STEPPERS, TRAIN_STEPPERS, save_model
from utils import load_config

NOW = datetime.now().strftime("%y%m%d_%H%M%S")


def format_time(seconds):
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
    model: nnx.Module,
    train_step: Callable,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    eps: float = 1e-8,
    loss_labels: Optional[list[str]] = None,
    run: str = "",
    config_string: str = "",
    local: bool = False,
    pretrain: bool = False,
    *,
    key: jax.Array,
):
    """
    Train a model given a dataset loader.

    Parameters
    ----------
    model : nnx.Module
        The model to train
    train_loader : DataLoader
        A train_loader for the given dataset
    val_loader : DataLoader
        A val_loader for the given dataset
    cfg : TrainerConfig
        Config specifying epochs and learning rate
    key : Array
        PRNG key
    loss_labels : list[str] = None
        The labels for each loss term
    run : str = ""
        The name of the current run
    config_string : str = ""
        The model config
    local : bool = False
        Whether the program should output information
    """
    if local:
        trackio.init(
            project="Interpretable Diffusion",
            name=run,
            config={
                "epochs": epochs,
                "learning_rate": learning_rate,
                # TODO: pass batch_size here
                "batch_size": next(iter(train_loader))[0].shape[0],
            },
        )

    optimizer = nnx.Optimizer(model, optax.adam(learning_rate, eps=eps), wrt=nnx.Param)

    final_loss = 0
    best_loss = float("inf")
    best_epoch = 0
    print("\nStarting training steps\n")

    pbar = tqdm(range(epochs)) if local else range(epochs)

    start = time.perf_counter()
    for epoch in pbar:
        running_loss = defaultdict(float)

        sub_pbar = (
            tqdm(train_loader, desc="Iterating batches", leave=False)
            if local
            else train_loader
        )
        for batch, _ in sub_pbar:
            key, train_key = split(key, num=2)
            losses = train_step(
                model=model, data=batch, optimizer=optimizer, key=train_key
            )

            assert len(losses.shape) == 1, "Only vector loss outputs can be processed"

            if local:
                if not loss_labels:
                    loss_labels = ["loss"] + [
                        f"sub_loss_{i + 1}" for i in range(len(losses) - 1)
                    ]

                for loss_label, loss in zip(loss_labels, losses):
                    running_loss[loss_label] += loss.item() / len(train_loader)

        if local:
            trackio.log(running_loss)

        curr_loss = running_loss["loss"]
        if best_loss >= curr_loss:
            best_loss = curr_loss
            best_epoch = epoch
        elif not local:
            print(
                f"Loss surpassed best loss: {curr_loss:.2E}. Best loss: {best_loss:.2E}."
            )

        if local:
            assert isinstance(pbar, tqdm)  # For type hints
            pbar.set_description(
                f"{'Pre' if pretrain else ''}Training, loss={running_loss['loss']:.2E} MAE={running_loss['mae']:.2E}"
            )

        final_loss = running_loss["loss"]
    end = time.perf_counter()

    key, eval_key = split(key, num=2)
    mae = evaluate(model, val_loader, leave=False, key=eval_key)

    if local:
        report = f"""# Training report
    - Final MAE: {mae:.2E}
    - Final loss: {final_loss:.2E}
    - Train time: {format_time(end - start)}

    ## Model config
    ```yaml
    {config_string}
    ```
        """
        trackio.log({"training_report": trackio.Markdown(report)})

        trackio.finish()
    else:
        print(
            f"""
# Training Report
Final MAE: {mae:.2E}
Final loss: {final_loss:.2E}
Best loss: {best_loss:.2E}
Best epoch: {best_epoch}
Train time: {format_time(end - start)}
"""
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", help="path to config file", required=True)
    parser.add_argument("-s", "--seed", type=int, help="seed for the run", default=0)
    parser.add_argument("--local", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    print("\nConfig loaded...")

    key = PRNGKey(args.seed)
    rngs = nnx.Rngs(args.seed)

    train_loader, val_loader, _ = DATASETS[config["dataset"]](
        **config["dataset_config"]
    )

    config_string = yaml.dump(config["dataset_config"], indent=4, sort_keys=False)

    model = MODELS[config["model"]](**config["model_config"], rngs=rngs)

    pretrain_step = PRETRAIN_STEPPERS.get(config["model"])
    train_step = TRAIN_STEPPERS[config["model"]]
    loss_labels = LOSS_LABELS[config["model"]]

    run_name = "_".join(
        [
            config["model"],
            config["dataset"],
            config["dataset_config"].get("system", ""),
            datetime.now().strftime("%m%d-%H%M"),
        ]
    )
    run_name = run_name.replace("__", "_")

    print("\nSetup complete...")

    if pretrain_step is not None:
        key, subkey = split(key)
        train(
            model,
            pretrain_step,
            train_loader,
            **config["trainer_config"],
            loss_labels=["loss"],
            run="pretrain_" + run_name,
            config_string=config_string,
            local=args.local,
            pretrain=True,
            key=subkey,
        )

    key, subkey = split(key)
    train(
        model,
        train_step,
        train_loader,
        val_loader,
        **config["trainer_config"],
        loss_labels=loss_labels,
        run=run_name,
        config_string=config_string,
        local=args.local,
        pretrain=False,
        key=subkey,
    )

    save_model(model, config["checkpoint_dir"])
