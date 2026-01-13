import os
import yaml
from argparse import ArgumentParser
from datetime import datetime

from jax.random import PRNGKey, split
from flax import nnx
import orbax.checkpoint as ocp

import matplotlib.pyplot as plt

from models import load_model, make_model, get_trainer
from datasets import get_dataset
from models.vae import VAE
from utils.config.trainer import TrainerConfig


if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '-c', '--config',
        help="path to config file",
        required=True
    )
    args = parser.parse_args()

    try:
        with open(args.config, "r") as config_file:
            config = yaml.safe_load(config_file)
    except FileNotFoundError:
        print(f"Provided config file, {args.config} not found")
        quit()
    except yaml.YAMLError:
        print("Error processing the config file. Verify the YAML formatting")
        quit()

    key = PRNGKey(0)
    rngs = nnx.Rngs(0)

    train_loader, test_loader = get_dataset(
        config.get('dataset'), config.get('dataset_config')
    )

    model = make_model(
        config.get('model'), config.get('model_config'), rngs=rngs
    )

    train = get_trainer(config.get('model'))
    key, subkey = split(key)
    train(
        model, train_loader,
        TrainerConfig(**config.get('trainer_config')),
        subkey
    )

    if not os.path.isdir(config.get('checkpoint_dir')):
        os.makedirs(config.get('checkpoint_dir'))

    _, state = nnx.split(model)
    now = datetime.now().strftime("%y%m%d_%H%M%S")

    # Async used to verify the full save occurs
    checkpointer = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
    checkpointer.save(os.path.join(
        os.getcwd(), config.get('checkpoint_dir'), now
    ), state)

    checkpointer.wait_until_finished()

    loaded_model = load_model(
        config.get('model'),
        config.get('model_config'),
        os.path.join(os.getcwd(), config.get('checkpoint_dir'), "260113_105710"),
        rngs=nnx.Rngs(1)
    )

    assert isinstance(loaded_model, VAE)
    key, enc_key, dec_key = split(key, num=3)
    latents = loaded_model.encode_and_sample(
        next(iter(test_loader))[0], key=enc_key)
    imgs = loaded_model.decode(latents)

    fig, ax = plt.subplots(1, 1, figsize=(12,5))
    fig.tight_layout()

    ax.set_xticks([])
    ax.set_yticks([])

    imgs = imgs[:100].reshape(5,20,28,28,1)
    grid = imgs.transpose(0,2,1,3,4).reshape(5*28,20*28,1)
    ax.imshow(grid, cmap="gray")
    plt.savefig("test.png", format="png")

    imgs = next(iter(test_loader))[0]
    imgs = imgs[:100].reshape(5,20,28,28,1)
    grid = imgs.transpose(0,2,1,3,4).reshape(5*28,20*28,1)
    ax.imshow(grid, cmap="gray")
    plt.savefig("ref.png", format="png")
