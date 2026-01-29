from dataclasses import dataclass

from pydantic import validate_arguments

from .modules import ModuleConfig

ModelArch = list[ModuleConfig]


@validate_arguments
@dataclass
class VAEConfig:
    encoder: ModelArch
    decoder: ModelArch
    latent_dim: int = 2
    channel_start: int = 6
    initial_shape: int = 7
    output_shape: int = 28
    eps: float = 1e-6
