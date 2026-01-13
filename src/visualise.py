import os
from datetime import datetime
from argparse import ArgumentParser

from flax import nnx
from jax.random import PRNGKey, split

import matplotlib.pyplot as plt

from datasets import get_dataset
from models import load_model
from models.vae import VAE
from utils.config import load_config


NOW = datetime.now().strftime("%y%m%d_%H%M%S")


if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '-c', '--config', help="path to config file", required=True
    )
    parser.add_argument(
        '-o', '--output', help="output folder for images", required=True
    )
    parser.add_argument(
        '--format', help="output image format", default="png"
    )
    parser.add_argument(
        '--checkpoint',
        help="checkpoint file in YYMMDD_HHMMSS format", default="latest"
    )
    args = parser.parse_args()
    config = load_config(args.config)

    rngs = nnx.Rngs(0)
    key = PRNGKey(0)

    model = load_model(
        config.get('model'), config.get('model_config'),
        os.path.join(
            os.getcwd(), config.get('checkpoint_dir'), args.checkpoint),
        rngs=rngs
    )

    _, test_loader = get_dataset(
        config.get('dataset'), config.get('dataset_config')
    )

    assert isinstance(model, VAE), "Only implemented for VAE"

    key, enc_key, dec_key = split(key, num=3)
    latents = model.encode_and_sample(
        next(iter(test_loader))[0], key=enc_key)
    imgs = model.decode(latents)

    fig, ax = plt.subplots(1, 1, figsize=(12,5))
    fig.tight_layout()

    ax.set_xticks([])
    ax.set_yticks([])

    imgs = imgs[:100].reshape(5,20,28,28,1)
    grid = imgs.transpose(0,2,1,3,4).reshape(5*28,20*28,1)
    ax.imshow(grid, cmap="gray")

    if not os.path.isdir(args.output):
        os.makedirs(args.output)

    plt.savefig(
        os.path.join(args.output, f"{NOW}.{args.format}"), format=args.format
    )
    plt.savefig(
        os.path.join(args.output, f"latest.{args.format}"), format=args.format
    )
