from pydantic.dataclasses import dataclass


@dataclass
class DatasetConfig:
    dir: str
    download: bool = False
    batch_size: int = 128


@dataclass
class PiVAEConfig:
    dir: str
    system: str
    n_samples: int = 3500
    test_split: int = 500
    validation_split: int = 1000
    train_split: int = 1000
    timesteps: int = 50
    batch_size: int = 128
