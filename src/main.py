from jax.random import PRNGKey, split
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from flax import nnx
import jax.numpy as jnp

import matplotlib.pyplot as plt

from models import VAE, train_vae


def preprocess_images(images):
    expanded = jnp.expand_dims(jnp.asarray(images), -1)

    return expanded / jnp.max(expanded)


def preprocess_targets(targets):
    return jnp.asarray(targets)


if __name__=='__main__':
    key = PRNGKey(0)

    digits = load_digits()
    splits = train_test_split(
        digits.images, digits.target, random_state=0
    )
    images_train, images_test = map(
        preprocess_images, splits[:2]
    )
    label_train, label_test = map(
        preprocess_targets, splits[2:]
    )

    latent_dim = 10
    model = VAE(latent_dim, rngs=nnx.Rngs(0))

    key, subkey = split(key)
    train_vae(model, images_train, key=subkey)

    key, subkey = split(key)
    enc_key, dec_key = split(subkey, num=2)
    latents = model.encode_and_sample(
        images_train[:10], key=enc_key)
    imgs = model.decode(latents)

    _, ax = plt.subplots(1, 1, figsize=(12,5))
    ax.set_xticks([])
    ax.set_yticks([])

    imgs = imgs.reshape(2,5,8,8,1)
    grid = imgs.transpose(0,2,1,3,4).reshape(2*8,5*8,1)
    ax.imshow(grid, cmap="gray")
    plt.savefig("test.png", format="png")

    imgs = images_train[:10].reshape(2,5,8,8,1)
    grid = imgs.transpose(0,2,1,3,4).reshape(2*8,5*8,1)
    ax.imshow(grid, cmap="gray")
    plt.savefig("ref.png", format="png")
