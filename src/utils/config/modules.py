from pydantic.dataclasses import dataclass
from typing import Optional


@dataclass
class ConvConfig:
    in_features: int
    out_features: int
    kernel_size: tuple[int, int]
    strides: Optional[tuple[int, int]] = None
    padding: str = "valid"
    act: str = "relu"


@dataclass
class LinearConfig:
    in_features: int
    out_features: Optional[int] = None
    act: str = "relu"


@dataclass
class MaxPoolConfig:
    window_shape: tuple[int, int]
    strides: Optional[tuple[int, int]] = None


@dataclass
class UpSampleConfig:
    shape: tuple[int, int]
    method: str


@dataclass
class ModuleConfig:
    type: str
    config: dict[str, str | int | list[int]]
