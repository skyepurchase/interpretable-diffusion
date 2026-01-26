import os
from argparse import ArgumentParser
from datetime import datetime

from jax.random import PRNGKey, split
from flax import nnx
import orbax.checkpoint as ocp

from models import make_model, get_trainer
from datasets import get_dataset
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

    train_loader, test_loader = get_dataset(
        config.get("dataset"), config.get("dataset_config")
    )

    model = make_model(config.get("model"), config.get("model_config"), rngs=rngs)

    train = get_trainer(config.get("model"))
    key, subkey = split(key)
    train(model, train_loader, TrainerConfig(**config.get("trainer_config")), subkey)

    if not os.path.isdir(config.get("checkpoint_dir")):
        os.makedirs(config.get("checkpoint_dir"))

    _, state = nnx.split(model)

    # Async used to verify the full save occurs
    checkpointer = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
    checkpointer.save(
        os.path.join(os.getcwd(), config.get("checkpoint_dir"), NOW), state
    )
    checkpointer.save(
        os.path.join(os.getcwd(), config.get("checkpoint_dir"), "latest"), state
    )

    checkpointer.wait_until_finished()
