import copy
from argparse import ArgumentParser
from datetime import datetime

import jax
from flax import nnx

from eval import evaluate
from main import format_time
from models import save_model
from train import train
from utils import load_config


def sweep_conditions(
    model_name: str,
    base_config: dict,
    conditions: list[int],
    checkpoint_dir: str,
    run_base: str,
    progress: bool = False,
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
    checkpoint_dir : str
        Directory to save checkpoints
    run_base : str
        Base run name
    rngs : nnx.Rngs
        Flax random number generators for model initialization
    progress : bool
        Whether to show progress bars and log metrics to trackio
    """
    seeds = [354916, 209358, 902408, 881891, 198910]
    keys = [jax.random.key(seed) for seed in seeds]
    rngs = [nnx.Rngs(seed) for seed in seeds]

    if progress:
        try:
            import trackio
        except ImportError as err:
            raise SystemExit(
                "trackio is not installed; install the dashboard extra with "
                "`uv sync --extra dashboard`"
            ) from err

    for condition in conditions:
        print(f"\nTRAINING WITH CONDITION PERIOD k={condition}\n")

        config = copy.deepcopy(base_config)
        config["model_config"]["condition"] = condition

        for key, rng, seed in zip(keys, rngs, seeds):
            stamp = datetime.now().strftime("%m%d-%H%M")
            run_name = f"{run_base}_{condition}_{seed}"

            if progress:
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

            train_key, eval_key = jax.random.split(key, num=2)

            result = train(config, rngs=rng, progress=progress, key=train_key)

            print("\nSaving model...")
            save_model(result.model, checkpoint_dir, stamp)
            print("Saved.")

            print("\nEvaluating...")
            forecast_mae, coeff_mae = evaluate(
                result.model, result.val_loader, args.progress, key=eval_key
            )

            print(
                f"""\nCondition {condition} complete:
Mean-absolute-error={forecast_mae}
Coefficient-error={coeff_mae}
Final-loss={result.final_loss:.2E}
Best-loss={result.best_loss:.2E}
Best-epoch={result.best_epoch}
Train-time={format_time(result.train_time)}
Model={config["model"]}
Dataset={config["dataset"]}"""
            )

            if progress:
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
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    print("\nConfig loaded.")

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
        checkpoint_dir=config["checkpoint_dir"],
        run_base=run_name,
        progress=args.progress,
    )
