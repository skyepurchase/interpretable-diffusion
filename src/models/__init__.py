import os
import shutil
from datetime import datetime

import orbax.checkpoint as ocp
from flax import nnx
from flax.nnx import Rngs

from models.new_paid import NewPaid
from models.paid import PAID
from models.phys_vae import PhysVAE
from models.train_ddim import train_step as ddim_step
from models.train_new_paid import pretrain_step as new_paid_pretrain_step
from models.train_new_paid import train_step as new_paid_step
from models.train_paid import train_step as paid_step
from models.train_phys_vae import train_step as phys_vae_step
from models.train_vae import train_step as vae_step

from .ddim import DDIM
from .vae import VAE

MODELS = {
    "vae": VAE,
    "ddim": DDIM,
    "phys_vae": PhysVAE,
    "paid": PAID,
    "new_paid": NewPaid,
}

TRAIN_STEPPERS: dict = {
    "ddim": ddim_step,
    "vae": vae_step,
    "phys_vae": phys_vae_step,
    "paid": paid_step,
    "new_paid": new_paid_step,
}

PRETRAIN_STEPPERS: dict = {
    "new_paid": new_paid_pretrain_step,
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
        "reconstruction",
        "physics",
        "discrepancy",
        "augmentation",
    ],
    "new_paid": [
        "loss",
        "reconstruction",
        "physics",
        "augmenter",
    ],
}


def load_model(name: str, cfg, checkpoint, rngs: Rngs) -> nnx.Module:
    """Load model (specified by the config) from a specific checkpoint"""
    model = MODELS[name](**cfg, rngs=rngs)
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
