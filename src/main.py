from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

import optax
from flax import nnx
import jax.numpy as jnp

from evaluate import accuracy
from models.vae import VAE
from train import train

digits = load_digits()
splits = train_test_split(
    digits.images, digits.target, random_state=0
)
images_train, images_test, label_train, label_test = map(
    jnp.asarray, splits
)

model = VAE(rngs=nnx.Rngs(0))
optimizer = nnx.ModelAndOptimizer(
    model, optax.sgd(learning_rate=0.05)
)
train(
    model,
    optimizer,
    images_train,
    label_train,
    images_test,
    label_test
)

accuracy(model, images_test, label_test)
