import copy
from argparse import ArgumentParser
from datetime import datetime

import jax
import trackio
from flax import nnx
from jaxtyping import PRNGKeyArray
from torch.utils.data import DataLoader

from datasets import DATASETS
from models import save_model
from train import train
from utils import load_config


def sweep_conditions(
    model_name: str,
    base_config: dict,
    conditions: list[int],
    train_loader: DataLoader,
    val_loader: DataLoader,
    checkpoint_dir: str,
    run_base: str,
    rngs: nnx.Rngs,
    local: bool = False,
    *,
    key: PRNGKeyArray,
):
    """
    Sweep over different condition values.

    Parameters
    ----------
    model_name : str
        Name of the model
    base_config : dict
        Base model configuration
    conditions : list[int]
        List of condition values to try
    train_loader : DataLoader
        Training data loader
    val_loader : DataLoader
        Validation data loader
    checkpoint_dir : str
        Directory to save checkpoints
    run_base : str
        Base run name
    rngs : nnx.Rngs
        Flax random number generators for model initialization
    local : bool
        Whether to output information locally
    key : PRNGKeyArray
        PRNG key for reproducibility during training
    """
    for condition in conditions:
        print(f"\nTRAINING WITH CONDITION PERIOD k={condition}\n")

        # Create config for this condition
        config = copy.deepcopy(base_config)
        config["model_config"]["condition"] = condition

        # Build run name with condition
        stamp = datetime.now().strftime("%m%d-%H%M")
        run_name = f"{run_base}_{stamp}_{condition}"

        # Initialize trackio for this condition
        trackio.init(
            project="Interpretable Diffusion",
            name=run_name,
            config={
                "condition": condition,
                "model": model_name,
                "dataset": config.get("dataset", "pivae"),
                **config.get("trainer_config", {}),
            },
        )

        # Split key for this condition
        key, subkey = jax.random.split(key)

        # Run training with rngs
        result = train(config, rngs=rngs, local=local, key=subkey)

        # Save model
        print("\nSaving model...")
        save_model(result.model, checkpoint_dir, stamp)
        print("Saved.")

        print(
            f"\nCondition {condition} complete:"
            f" Final loss: {result.final_loss:.2E}"
            f" Best loss: {result.best_loss:.2E}"
            f" Best epoch: {result.best_epoch}"
            f" Train time: {result.train_time:.2f}s"
        )

        trackio.finish()


if __name__ == "__main__":
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
        base_config=config,
        conditions=args.conditions,
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_dir=config["checkpoint_dir"],
        run_base=run_name,
        rngs=rngs,
        local=args.local,
        key=key,
    )
