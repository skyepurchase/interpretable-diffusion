from argparse import ArgumentParser
from datetime import datetime

import jax
import trackio
from flax import nnx
from jax.random import split

from eval import EVALUATORS
from models import save_model
from train import train
from utils import load_config


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


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", help="path to config file", required=True)
    parser.add_argument("-s", "--seed", type=int, help="seed for the run", default=0)
    parser.add_argument("--local", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    print("\nConfig loaded.")

    # Own the seed and initialize rngs
    key = jax.random.key(args.seed)
    rngs = nnx.Rngs(args.seed)

    # Compute run name and stamp
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

    # Split key into train and eval keys
    train_key, eval_key = split(key, num=2)

    # Initialize trackio once
    trackio.init(
        project="Interpretable Diffusion",
        name=run_name,
        config={
            "model": config["model"],
            "dataset": config["dataset"],
            **config.get("trainer_config", {}),
            **config.get("model_config", {}),
        },
    )

    print("\nSetup complete. Starting training...")

    # Train with rngs and train_key
    result = train(config, rngs=rngs, local=args.local, key=train_key)

    # Save model
    print("\nSaving model...")
    save_model(result.model, config["checkpoint_dir"], stamp)
    print("Saved.")

    # Evaluate
    print("\nEvaluating...")
    if config["model"] in EVALUATORS:
        forecast_mae, coeff_mae = EVALUATORS[config["model"]](
            result.model, result.val_loader, args.local, key=eval_key
        )
    else:
        print(f"Warning: {config['model']} is not supported for evaluation")
        forecast_mae = None
        coeff_mae = None

    # Build and log markdown report
    report = f"""# Training Report

## Metrics
- Mean absolute error: {forecast_mae:.2E}
- Coefficient error: {coeff_mae:.2E}
- Final loss: {result.final_loss:.2E}
- Best loss: {result.best_loss:.2E}
- Best epoch: {result.best_epoch}
- Train time: {format_time(result.train_time)}

## Config
- Model: {config["model"]}
- Dataset: {config["dataset"]}
"""

    trackio.log({"training_report": trackio.Markdown(report)})

    # Print summary when not local
    if not args.local:
        print(
            f"""\n# Training Report
- Mean absolute error: {forecast_mae:.2E}
- Coefficient error: {coeff_mae:.2E}
- Final loss: {result.final_loss:.2E}
- Best loss: {result.best_loss:.2E}
- Best epoch: {result.best_epoch}
- Train time: {format_time(result.train_time)}
"""
        )

    trackio.finish()


if __name__ == "__main__":
    main()
