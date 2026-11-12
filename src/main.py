from argparse import ArgumentParser
from datetime import datetime

import jax
from flax import nnx
from jax.random import split

from eval import evaluate
from models import save_model
from train import train
from utils import load_config
from visualise import visualise


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
    parser.add_argument("-o", "--output", help="path to config folder", required=True)
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

    train_key, eval_key, vis_key = split(key, num=3)

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
                "model": config["model"],
                "dataset": config["dataset"],
                **config.get("trainer_config", {}),
                **config.get("model_config", {}),
            },
        )

    print("\nSetup complete. Starting training...")

    result = train(config, rngs=rngs, progress=args.progress, key=train_key)

    print("\nSaving model...")
    save_model(result.model, config["checkpoint_dir"], stamp)
    print("Saved.")

    print("\nEvaluating...")
    forecast_mae, coeff_mae = evaluate(
        result.model, result.test_loader, args.progress, key=eval_key
    )

    print("\nVisualising...")
    visualise(
        result.model,
        result.test_loader,
        folder=args.output,
        length=config["dataset_config"]["length"],
        stamp=stamp,
        num=3,
        key=vis_key,
    )

    print(
        f"""\nMean-absolute-error={forecast_mae}
Coefficient-error={coeff_mae}
Final-loss={result.final_loss:.2E}
Best-loss={result.best_loss:.2E}
Best-epoch={result.best_epoch}
Train-time={format_time(result.train_time)}
Model={config["model"]}
Dataset={config["dataset"]}
Stamp={stamp}"""
    )

    if args.progress:
        trackio.finish()


if __name__ == "__main__":
    main()
