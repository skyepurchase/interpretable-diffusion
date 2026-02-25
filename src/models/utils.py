import os
import shutil
from datetime import datetime
from typing import Callable

import jax
import orbax.checkpoint as ocp
from flax import nnx
from flax.nnx import Rngs
from torch.utils.data import DataLoader

from models.phys_vae import PhysVAE
from models.train_ddim import train_ddim
from models.train_phys_vae import train_phys_vae
from models.train_vae import train_vae
from utils.config.models import DDIMConfig, PhysVAEConfig, VAEConfig
from utils.config.trainer import TrainerConfig

from .ddim import DDIM
from .vae import VAE

MODELS = {
    "vae": (VAE, VAEConfig),
    "ddim": (DDIM, DDIMConfig),
    "phys_vae": (PhysVAE, PhysVAEConfig),
}

TRAINERS = {
    "vae": train_vae,
    "ddim": train_ddim,
    "phys_vae": train_phys_vae,
}

TrainerFn = Callable[[nnx.Module, DataLoader, TrainerConfig, jax.Array], None]


def make_model(name: str, cfg, rngs: Rngs) -> nnx.Module:
    """Create a model object based on the name and assorted config"""
    if name in MODELS:
        model, config = MODELS[name]

        return model(config(**cfg), rngs)

    raise NotImplementedError(f"Model {name} not implemented")


def load_model(name: str, cfg, checkpoint, rngs: Rngs) -> nnx.Module:
    """Load model (specified by the config) from a specific checkpoint"""
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
        os.path.join(os.getcwd(), folder, datetime.now().strftime("%y%m%d_%H%M")), state
    )

    latest_path = os.path.join(os.getcwd(), folder, "latest")
    if os.path.isdir(latest_path):
        shutil.rmtree(latest_path)

    checkpointer.save(latest_path, state)

    checkpointer.wait_until_finished()


def get_trainer(
    name: str,
) -> TrainerFn:
    """Return the relevant trainer function for the desired model"""
    if name in TRAINERS:
        return TRAINERS[name]

    raise NotImplementedError(f"Trainer for {name} not implemented")
