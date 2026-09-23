from jaxtyping import PRNGKeyArray
from torch.utils.data import DataLoader

from datasets.pivae import get_pivae


def get_burgers(
    dir: str,
    system: str,
    batch_size: int = 128,
    n_samples: int = 3500,
    test_split: int = 500,
    validation_split: int = 1000,
    train_split: int = 1000,
    t_grid: int = 128,
    x_grid: int = 128,
    retain_dim: bool = False,
    *,
    key: PRNGKeyArray,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Wrapper for PiVAE"""
    return get_pivae(
        dir,
        "burgers",
        batch_size=batch_size,
        n_samples=n_samples,
        test_split=test_split,
        validation_split=validation_split,
        train_split=train_split,
        t_grid=t_grid,
        x_grid=x_grid,
        retain_dim=retain_dim,
        key=key,
    )
