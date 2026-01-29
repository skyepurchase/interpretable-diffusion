from argparse import ArgumentParser
from datetime import datetime

from flax import nnx
from jax.random import PRNGKey, split

from datasets import get_dataset
from models import get_trainer, make_model
from models.utils import save_model
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

    save_model(model, config.checkpoint_dir)
