import jax
from flax import nnx
from flax.nnx import Rngs
import orbax.checkpoint as ocp
from torch.utils.data import DataLoader

from utils.config.models import VAEConfig
from utils.config.trainer import TrainerConfig

from .vae import VAE
from .train_vae import train_vae

from typing import Callable


MODELS = {
    "vae": (VAE, VAEConfig),
}

TRAINERS = {"vae": train_vae}


def make_model(name: str, cfg, rngs: Rngs) -> nnx.Module:
    if name in MODELS:
        model, config = MODELS[name]
        return model(config(**cfg), rngs)

    raise ValueError(f"Model {name} not found")


def load_model(name: str, cfg, checkpoint, rngs: Rngs) -> nnx.Module:
    model = make_model(name, cfg, rngs)
    graph, state = nnx.split(model)

    checkpointer = ocp.StandardCheckpointer()
    checkpoint_state = checkpointer.restore(checkpoint, state)

    return nnx.merge(graph, checkpoint_state)


def get_trainer(
    name: str,
) -> Callable[[nnx.Module, DataLoader, TrainerConfig, jax.Array], None]:
    if name in TRAINERS:
        return TRAINERS[name]

    raise ValueError(f"Trainer for {name} not found")


__all__ = [
    "make_model",
    "load_model",
    "get_trainer",
]
