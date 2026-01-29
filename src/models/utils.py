import os
from datetime import datetime
from typing import Callable

import jax
import orbax.checkpoint as ocp
from flax import nnx
from flax.nnx import Rngs
from torch.utils.data import DataLoader

from utils.config.models import VAEConfig
from utils.config.trainer import TrainerConfig

from .train_vae import train_vae
from .vae import VAE

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


def save_model(model: nnx.Module, folder: str) -> None:
    """Save the given model in the specified folder."""
    if not os.path.isdir(folder):
        os.makedirs(folder)

    _, state = nnx.split(model)

    # Async used to verify the full save occurs
    checkpointer = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
    checkpointer.save(
        os.path.join(folder, datetime.now().strftime("%y%m%d_%H%M%S")), state
    )
    checkpointer.save(os.path.join(folder, "latest"), state)

    checkpointer.wait_until_finished()


def get_trainer(
    name: str,
) -> Callable[[nnx.Module, DataLoader, TrainerConfig, jax.Array], None]:
    if name in TRAINERS:
        return TRAINERS[name]

    raise ValueError(f"Trainer for {name} not found")
