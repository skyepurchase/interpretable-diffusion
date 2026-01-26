from jax.image import ResizeMethod
from pydantic.dataclasses import dataclass

from .modules import ModuleConfig


ModelArch = list[ModuleConfig]


@dataclass
class VAEConfig:
    encoder: ModelArch
    decoder: ModelArch
    latent_dim: int = 2
    channel_start: int = 6
    initial_shape: int = 7
    output_shape: int = 28
    eps: float = 1e-6


@dataclass
class TUNetConfig:
    in_features: int
    out_features: int
    time_features: int
    feature_multiples: list[int]
    act: str
    resize_method: str | ResizeMethod = "bilinear"


@dataclass
class DenoiserConfig:
    model: str
    config: dict


@dataclass
class DDIMConfig:
    denoiser: DenoiserConfig
    time_embedding: int
    time_features: int
    beta_schedule: str
    beta_start: float
    beta_end: float
    num_timesteps: int
    act: str
