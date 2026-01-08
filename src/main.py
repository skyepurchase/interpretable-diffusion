from jax.random import PRNGKey, split
from flax import nnx

import matplotlib.pyplot as plt

from models import VAE, train_vae
from datasets import get_MNIST


if __name__=='__main__':
    key = PRNGKey(0)

    train_loader, test_loader = get_MNIST('./data/MNIST')

    latent_dim = 2
    model = VAE(latent_dim, rngs=nnx.Rngs(0))

    key, subkey = split(key)
    train_vae(model, train_loader, key=subkey, epochs=15)

    key, enc_key, dec_key = split(subkey, num=3)
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
    plt.savefig("test.png", format="png")

    imgs = next(iter(test_loader))[0]
    imgs = imgs[:100].reshape(5,20,28,28,1)
    grid = imgs.transpose(0,2,1,3,4).reshape(5*28,20*28,1)
    ax.imshow(grid, cmap="gray")
    plt.savefig("ref.png", format="png")
