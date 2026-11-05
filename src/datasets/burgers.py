from jaxtyping import PRNGKeyArray

from datasets.loader import ArrayLoader
from datasets.pivae import get_pivae


def get_burgers(
    dir: str,
    system: str,
    length: float = 1.0,
    time: float = 0.1,
    phys_param_ranges: list | None = None,
    n_samples: int = 3000,
    test_split: int = 1000,
    validation_split: int = 500,
    train_split: int = 1000,
    t_grid: int = 128,
    x_grid: int = 128,
    batch_size: int = 128,
    retain_dim: bool = False,
    *,
    key: PRNGKeyArray,
) -> tuple[ArrayLoader, ArrayLoader, ArrayLoader]:
    """Wrapper for PiVAE"""
    return get_pivae(
        dir,
        "burgers",
        length=length,
        time=time,
        phys_param_ranges=phys_param_ranges,
        n_samples=n_samples,
        test_split=test_split,
        validation_split=validation_split,
        train_split=train_split,
        t_grid=t_grid,
        x_grid=x_grid,
        batch_size=batch_size,
        retain_dim=retain_dim,
        key=key,
    )
