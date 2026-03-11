from pydantic.dataclasses import dataclass


@dataclass
class TrainerConfig:
    lr: float = 1e-3
    eps: float = 1e-8
    epochs: int = 15
    snapshot_freq: int = 1
