from argparse import ArgumentParser
from collections import defaultdict
from datetime import datetime

import jax
import optax
import trackio
from flax import nnx
from jax.random import PRNGKey, split
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import get_dataset
from models import get_trainer, make_model
from models.utils import save_model
from utils.config import load_config
from utils.config.trainer import TrainerConfig
from utils.misc import TrainStep

NOW = datetime.now().strftime("%y%m%d_%H%M%S")


def train(
    model: nnx.Module,
    train_step: TrainStep,
    dataloader: DataLoader,
    cfg: TrainerConfig,
    key: jax.Array,
    run: str = "",
):
    """
    Train a model given a dataset loader.

    Parameters
    ----------
    model : nnx.Module
        The model to train
    dataloader : DataLoader
        A dataloader for the given dataset
    cfg : TrainerConfig
        Config specifying epochs and learning rate
    key : Array
        PRNG key
    run : str
        The name of the current run
    """
    trackio.init(
        project="Interpretable Diffusion",
        name=run,
        config={
            "epochs": cfg.epochs,
            "learning_rate": cfg.lr,
            # TODO: pass batch_size here
            "batch_size": next(iter(dataloader))[0].shape[0],
        },
    )

    optimizer = nnx.Optimizer(model, optax.adam(cfg.lr, eps=cfg.eps), wrt=nnx.Param)

    pbar = tqdm(range(cfg.epochs), desc="Training")
    for _ in pbar:
        running_loss = defaultdict(float)

        for imgs, _ in tqdm(dataloader, desc="Iterating images", leave=False):
            key, train_key = split(key, num=2)
            losses = train_step(model, imgs, optimizer, key=train_key)

            assert len(losses.shape) == 1, "Only vector loss outputs can be processed"

            for i, value in enumerate(losses):
                running_loss[f"loss_{i}"] += value.item() / len(dataloader)

        trackio.log(running_loss)

        pbar.set_description(f"Training, loss={running_loss['loss_0']:.2f}")

    trackio.finish()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", help="path to config file", required=True)
    parser.add_argument("-s", "--seed", type=int, help="seed for the run", default=0)
    args = parser.parse_args()
    config = load_config(args.config)

    key = PRNGKey(args.seed)
    rngs = nnx.Rngs(args.seed)

    train_loader, val_loader, test_loader = get_dataset(
        config.dataset, config.dataset_config
    )

    model = make_model(config.model, config.model_config, rngs=rngs)

    train_step = get_trainer(config.model)
    key, subkey = split(key)

    run_name = "_".join([config.model, config.dataset, str(args.seed)])

    train(
        model,
        train_step,
        train_loader,
        TrainerConfig(**config.trainer_config),
        subkey,
        run=run_name,
    )

    save_model(model, config.checkpoint_dir)
