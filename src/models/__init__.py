import os
import shutil

import orbax.checkpoint as ocp
from flax import nnx
from flax.nnx import Rngs

from models.paid import PAID
from models.phys_vae import PhysVAE
from models.train_ddim import train_step as ddim_step
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
}

TRAIN_STEPPERS: dict = {
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
        "reconstruction",
        "physics",
        "discrepancy",
        "augmentation",
    ],
}


PHYSICS_MODELS = {"phys_vae", "paid"}


def make_model(name: str, cfg: dict, dataset_config: dict, rngs: Rngs) -> nnx.Module:
    """Build a model from its configuration.

    Physics models (phys_vae, paid) additionally receive the PDE system
    definition shared with data generation from dataset_config: system,
    length, time, x_grid, and t_grid.
    """
    if name in PHYSICS_MODELS:
        return MODELS[name](
            **cfg,
            system=dataset_config["system"],
            length=dataset_config["length"],
            time=dataset_config["time"],
            x_grid=dataset_config["x_grid"],
            t_grid=dataset_config["t_grid"],
            rngs=rngs,
        )
    return MODELS[name](**cfg, rngs=rngs)


def load_model(
    name: str, cfg: dict, dataset_config: dict, checkpoint, rngs: Rngs
) -> nnx.Module:
    """Load model (specified by the config) from a specific checkpoint"""
    model = make_model(name, cfg, dataset_config, rngs)
    graph, state = nnx.split(model)

    checkpointer = ocp.StandardCheckpointer()
    checkpoint_state = checkpointer.restore(checkpoint, state)

    return nnx.merge(graph, checkpoint_state)


def save_model(model: nnx.Module, folder: str, stamp: str) -> None:
    """Save the given model in the specified folder."""
    if not os.path.isdir(folder):
        os.makedirs(folder)

    _, state = nnx.split(model)

    # Async used to verify the full save occurs
    checkpointer = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())

    perm_path = os.path.join(os.getcwd(), folder, stamp)
    if os.path.isdir(perm_path):
        print(f"Unique stamp ({stamp}) already exists. Use a timestamp to avoid this.")
        print("Skipping permanent saving.")
    else:
        checkpointer.save(perm_path, state)

    latest_path = os.path.join(os.getcwd(), folder, "latest")
    if os.path.isdir(latest_path):
        shutil.rmtree(latest_path)

    checkpointer.save(latest_path, state)

    checkpointer.wait_until_finished()
