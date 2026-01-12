from flax.nnx import Rngs

from .vae import VAE, VAEConfig
from .train_vae import train_vae


MODELS = {
    'vae': (VAE, VAEConfig)
}


def make_model(name: str, cfg, rngs: Rngs):
    if name in MODELS:
        model, config = MODELS[name]
        return model(config(**cfg), rngs)

    raise ValueError(f"Model {name} not found")


__all__ = [
    'make_model',
    'train_vae',
]
