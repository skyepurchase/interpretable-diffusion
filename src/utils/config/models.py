from dataclasses import field
from typing import Optional

from pydantic.dataclasses import dataclass

from .modules import ModuleConfig

ModelArch = list[ModuleConfig]


@dataclass
class PhysFunc:
    latent_dim: int
    dims: list[int]
    act: str = "elu"
    residual: bool = False


@dataclass
class PhysEncoderConfig:
    aux_funcs: list[PhysFunc]
    unmixer: PhysFunc
    phys_funcs: list[PhysFunc]


@dataclass
class PhysDecoderConfig:
    pre_funcs: list[PhysFunc]
    post_funcs: list[PhysFunc]
    x_log_sigma: float = 0.01


@dataclass
class PhysVAEConfig:
    encoder: PhysEncoderConfig
    decoder: PhysDecoderConfig
    time_dim: int

    phys_param_ranges: list[tuple[float, float]]

    system: str
    integral_level: int
    timestep: float
    ode_solver: str

    alpha: float
    beta: float
    gamma: float

    length: float = 0.0
    num_grid_points: int = 1


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
class UNetConfig:
    in_features: int
    base_features: int
    feature_multiples: tuple[int, ...]
    num_blocks: int
    attention_resolutions: tuple[int, ...]
    attention_heads: int
    act: str
    dropout: bool
    dropout_rate: float


@dataclass
class DDIMConfig(UNetConfig):
    num_timesteps: int
    beta_schedule: str
    beta_start: float
    beta_end: float
    pad: list[tuple] = field(default_factory=lambda: [(0, 0), (0, 0)])


@dataclass
class PhysBlock:
    latent_dim: int
    in_features: int
    features: int
    act: str
    dropout: bool
    dropout_rate: float
    time_dim: Optional[int] = None


@dataclass
class UnmixerBlock:
    dims: list[int]
    act: str
    dropout: bool
    dropout_rate: float
    time_dim: Optional[int] = None


@dataclass
class PDEConfig:
    system: str
    phys_param_ranges: list[tuple[float, float]]
    time_dim: int
    timestep: float
    solver: str
    integral_level: int
    spatial_length: float = 0.0
    num_grid_points: int = 1


@dataclass
class PAIDConfig:
    unet: UNetConfig
    pde: PDEConfig

    aux_funcs: list[PhysBlock]
    unmixer: UnmixerBlock
    phys_funcs: list[PhysBlock]
    pre_funcs: list[PhysFunc]
    post_funcs: list[PhysBlock]

    num_timesteps: int
    beta_schedule: str
    beta_start: float
    beta_end: float

    alpha: float
    beta: float
    gamma: float

    condition: int = 5
    pad: list[tuple] = field(default_factory=lambda: [(0, 0), (0, 0)])
