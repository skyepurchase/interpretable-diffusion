from dataclasses import dataclass

from .models import ModelConfig


@dataclass
class Experiment:
    model: str
    model_config: ModelConfig


def parse_experiment(yaml):

