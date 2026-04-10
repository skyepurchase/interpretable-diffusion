import os
import shutil
from datetime import datetime
from typing import Optional

import orbax.checkpoint as ocp
from flax import nnx
from flax.nnx import Rngs

from models.paid import PAID
from models.phys_vae import PhysVAE
from models.train_ddim import train_step as ddim_step
from models.train_paid import train_step as paid_step
from models.train_phys_vae import train_step as phys_vae_step
from models.train_vae import train_step as vae_step
from utils.config.models import DDIMConfig, PAIDConfig, PhysVAEConfig, VAEConfig
from utils.misc import TrainStep

from .ddim import DDIM
from .vae import VAE

MODELS = {
    "vae": (VAE, VAEConfig),
    "ddim": (DDIM, DDIMConfig),
    "phys_vae": (PhysVAE, PhysVAEConfig),
    "paid": (PAID, PAIDConfig),
}

TRAIN_STEPPER: dict[str, TrainStep] = {
    "ddim": ddim_step,
    "vae": vae_step,
    "phys_vae": phys_vae_step,
    "paid": paid_step,
}

LOSS_LABELS: dict[str, list[str]] = {
    "ddim": ["loss"],
    "vae": ["loss"],
    "phys_vae": [
        "loss",
        "reconstruction",
        "kl",
        "discrepancy",
        "physics",
        "augmentation",
    ],
    "paid": [
        "loss",
        "estimation",
        "physics",
    ],
}


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
) -> tuple[TrainStep, Optional[list[str]]]:
    """Return the relevant trainer function for the desired model"""
    if name in TRAIN_STEPPER:
        return TRAIN_STEPPER[name], LOSS_LABELS[name]

    raise NotImplementedError(f"Trainer for {name} not implemented")


__all__ = [
    "make_model",
    "load_model",
    "get_trainer",
]
