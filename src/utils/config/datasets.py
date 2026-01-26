from pydantic.dataclasses import dataclass


@dataclass
class DatasetConfig:
    dir: str
    download: bool = False
    batch_size: int = 128
