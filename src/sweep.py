from typing import Callable, Optional

from flax.nnx import Rngs
from jaxtyping import PRNGKeyArray
from torch.utils.data import DataLoader

from models import MODELS, save_model
from train import train


def sweep_conditions(
    model_name: str,
    config: dict,
    conditions: list[int],
    train_step: Callable,
    train_loader: DataLoader,
    val_loader: DataLoader,
    checkpoint_dir: str,
    epochs: int,
    learning_rate: float,
    eps: float = 1e-8,
    loss_labels: Optional[list[str]] = None,
    run_base: str = "",
    config_string: str = "",
    local: bool = False,
    *,
    key: PRNGKeyArray,
    rngs: Rngs,
):
    for condition in conditions:
        print(f"\nTRAINING WITH CONDITION PERIOD k={condition}\n")

        config["condition"] = condition
        model = MODELS[model_name](**config, rngs=rngs)

        stamp = datetime.now().strftime("%m%d-%H%M")
        run = run_base + f"_{stamp}_{condition}"

        train(
            model,
            train_step,
            train_loader,
            val_loader,
            epochs,
            learning_rate,
            eps,
            loss_labels,
            run,
            config_string,
            local,
            pretrain=False,
            key=key,
        )

        print("\nSaving model...")
        save_model(model, checkpoint_dir, stamp)
        print("Saved.")


if __name__ == "__main__":
    from argparse import ArgumentParser
    from datetime import datetime

    import jax
    import yaml
    from flax import nnx

    from datasets import DATASETS
    from models import LOSS_LABELS, TRAIN_STEPPERS
    from utils import load_config

    parser = ArgumentParser()
    parser.add_argument("-c", "--config", help="path to config file", required=True)
    parser.add_argument(
        "--conditions",
        type=int,
        nargs="+",
        help="The condition periods to try",
        required=True,
    )
    parser.add_argument("-s", "--seed", type=int, help="seed for the run", default=0)
    parser.add_argument("--local", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    print("\nConfig loaded.")

    key = jax.random.key(args.seed)
    rngs = nnx.Rngs(args.seed)

    data_key, key = jax.random.split(key)
    train_loader, val_loader, _ = DATASETS[config["dataset"]](
        **config["dataset_config"], key=data_key
    )

    config_string = yaml.dump(config["dataset_config"], indent=4, sort_keys=False)

    train_step = TRAIN_STEPPERS[config["model"]]
    loss_labels = LOSS_LABELS[config["model"]]

    run_name = "_".join(
        [
            config["model"],
            config["dataset"],
            config["dataset_config"].get("system", ""),
        ]
    )
    run_name = run_name.replace("__", "_")

    print("\nSetup complete.")

    sweep_conditions(
        model_name=config["model"],
        config=config["model_config"],
        conditions=args.conditions,
        train_step=train_step,
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_dir=config["checkpoint_dir"],
        **config["trainer_config"],
        loss_labels=loss_labels,
        run_base=run_name,
        config_string=config_string,
        local=args.local,
        key=key,
        rngs=rngs,
    )
