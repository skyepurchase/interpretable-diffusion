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
    img_gen: bool = True


@dataclass
class DDIMConfig:
    in_features: int
    base_features: int
    feature_multiples: tuple[int, ...]
    num_blocks: int
    attention_resolutions: tuple[int, ...]
    attention_heads: int
    act: str
    dropout: bool
    dropout_rate: float
    num_timesteps: int
    beta_schedule: str
    beta_start: float
    beta_end: float
