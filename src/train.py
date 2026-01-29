import os
from argparse import ArgumentParser
from datetime import datetime

import orbax.checkpoint as ocp
from flax import nnx
from jax.random import PRNGKey, split

from datasets import get_dataset
from models import get_trainer, load_model, make_model
from utils.config import load_config
from utils.config.trainer import TrainerConfig

NOW = datetime.now().strftime("%y%m%d_%H%M%S")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", help="path to config file", required=True)
    args = parser.parse_args()
    config = load_config(args.config)

    key = PRNGKey(0)
    rngs = nnx.Rngs(0)

    train_loader, test_loader = get_dataset(config.dataset, config.dataset_config)

    model = make_model(config.model, config.model_config, rngs=rngs)

    train = get_trainer(config.model)
    key, subkey = split(key)
    train(model, train_loader, TrainerConfig(**config.trainer_config), subkey)

    if not os.path.isdir(config.checkpoint_dir):
        os.makedirs(config.checkpoint_dir)

    _, state = nnx.split(model)

    # Async used to verify the full save occurs
    checkpointer = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
    checkpointer.save(os.path.join(os.getcwd(), config.checkpoint_dir, NOW), state)
    checkpointer.save(os.path.join(os.getcwd(), config.checkpoint_dir, "latest"), state)

    checkpointer.wait_until_finished()

    loaded_model = load_model(
        config.model,
        config.model_config,
        os.path.join(os.getcwd(), config.checkpoint_dir, "260113_105710"),
        rngs=nnx.Rngs(1),
    )
