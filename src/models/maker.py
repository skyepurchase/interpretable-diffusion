import jax
from flax import nnx
from flax.nnx import Rngs
import orbax.checkpoint as ocp
from torch.utils.data import DataLoader

from utils.config.models import DDIMConfig, TUNetConfig, VAEConfig
from utils.config.trainer import TrainerConfig

from .vae import VAE
from .train_vae import train_vae
from .ddim import DDIM
from .train_ddim import train_ddim
from .unet import TUNet

from typing import Callable


MODELS = {"vae": (VAE, VAEConfig), "temporal-unet": (TUNet, TUNetConfig)}

TRAINERS = {"vae": train_vae, "ddim": train_ddim}


def _make_ddim(cfg: DDIMConfig, rngs: nnx.Rngs) -> nnx.Module:
    if cfg.denoiser.model in MODELS:
        model, config = MODELS[cfg.denoiser.model]
        denoiser = model(config(**cfg.denoiser.config), rngs)

        return DDIM(cfg, denoiser, rngs)

    raise NotImplementedError(
        f"DDIM denoiser model {cfg.denoiser.model} not implemented"
    )


MODELS["ddim"] = (_make_ddim, DDIMConfig)


def make_model(name: str, cfg, rngs: Rngs) -> nnx.Module:
    if name in MODELS:
        model, config = MODELS[name]
        return model(config(**cfg), rngs)

    raise NotImplementedError(f"Model {name} not implemented")


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

    raise NotImplementedError(f"Trainer for {name} not implemented")
