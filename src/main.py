import jax
from jax.random import PRNGKey, split
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

import optax
from flax import nnx
import jax.numpy as jnp

import matplotlib.pyplot as plt

from evaluate import accuracy
from models import VAE
from train import train


def preprocess_images(images):
    expanded = jnp.expand_dims(jnp.asarray(images), -1)
    #upscaled = jax.image.resize(expanded, shape=(expanded.shape[0], 28, 28, 1),method="bilinear")

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
    enc_key, dec_key = split(subkey, num=2)
    latents = model.encode(images_train[:10]).sample(seed=enc_key)

    _, ax = plt.subplots(1, 1, figsize=(5,5))
    ax.imshow(model.decode(latents)[0].sample(seed=dec_key))
    plt.savefig("test.png", format="png")

    # optimizer = nnx.ModelAndOptimizer(
    #     model, optax.sgd(learning_rate=0.05)
    # )
    # train(
    #     model,
    #     optimizer,
    #     images_train,
    #     label_train,
    #     images_test,
    #     label_test
    # )
    # 
    # accuracy(model, images_test, label_test)
