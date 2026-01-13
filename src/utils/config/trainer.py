from dataclasses import dataclass
from pydantic import validate_arguments


@validate_arguments
@dataclass
class TrainerConfig:
    lr: float = 1e-3
    epochs: int = 15
