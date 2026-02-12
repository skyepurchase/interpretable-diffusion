import os
from typing import Callable

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from datasets.utils import np_collate
from utils.config.datasets import DatasetConfig

PRECISION = 1000


class PiVAEDataset(Dataset):
    """A wrapper of the torch Dataset for the PiVAE datasets."""

    def __init__(self, series: NDArray) -> None:
        super().__init__()
        self.data = torch.tensor(series[:, :, 1:]).flatten(start_dim=1)
        self.timestep = torch.tensor(series[:, :, 0]).flatten(start_dim=1)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        """"""
        return self.data[index], self.timestep[index]


def make_ODE() -> Callable[[NDArray, NDArray], NDArray]:
    """
    Create the force damped pendulum from A. Kalousis & N. Takeishi. This randomly samples the constants omega, xi, phi, and A from the ODE and returns a function that given the timestep and state returns the change in state.

    Returns
    -------
    func : (t, y) -> dy
        The function that progresses y according to the ODE.
    """
    omega = (np.random.randint(0, PRECISION) * (3.14 - 0.785) / PRECISION) + 0.785
    xi = np.random.randint(0, PRECISION) * 8 / PRECISION
    phi = (np.random.randint(0, PRECISION) * (6.280 - 3.14) / PRECISION) + 3.14
    A = np.random.randint(0, PRECISION) * 40 / PRECISION

    def force_damped_pendulum(t: NDArray, y: NDArray) -> NDArray:
        theta, theta_dot = y

        dtheta = theta_dot  # first derivative is velocity
        dtheta_dot = (
            (A * omega**2 * np.cos(2 * np.pi * phi * t))
            - (xi * theta_dot)
            - (omega**2 * np.sin(theta))
        )  # second derivative according to force dampened pendulum

        return np.array([dtheta, dtheta_dot])

    return force_damped_pendulum


def generate_pendulum_data(n_samples: int = 3500) -> NDArray:
    """
    Using the force damped ODE above generate a set of randomly initialised samples.

    Parameters
    ----------
    n_samples : int = 3500
        The number of samples to generate

    Returns
    -------
    samples : (B,T,2)
        B samples with T timesteps including both the timestep and the pendulum angle.
    """
    steps = 50
    T = steps * 0.05
    t_span = (0, T)

    samples = np.empty((0, steps + 1, 2))
    for _ in tqdm(range(n_samples), desc="Generating samples"):
        y_0 = (np.random.randint(0, PRECISION) * (1.57 + 1.57) / PRECISION) - 1.57
        y0 = [y_0, 0]

        sol = solve_ivp(
            fun=make_ODE(),
            t_span=t_span,
            y0=y0,
            t_eval=np.linspace(0, T, steps + 1),
            method="DOP853",
            rtol=1e-3,
            atol=1e-6,
        )

        series = np.concatenate(
            [sol.t[:, None], np.expand_dims(sol.y[0], axis=1)], axis=1
        )
        samples = np.append(samples, series[None, :, :], axis=0)

    return samples


def get_pendulum(cfg: DatasetConfig) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create force damped pendulum train, val, and test dataloaders"""
    n_samples = 3500
    test_split = 500
    validation_split = 1000
    train_split = 1000
    assert test_split + validation_split + train_split <= n_samples

    if not all(
        [
            os.path.exists(os.path.join(cfg.dir, file))
            for file in ["test.npy", "val.npy", "train_pool.npy"]
        ]
    ):
        samples = generate_pendulum_data(n_samples)
        assert samples.shape == (n_samples, 51, 2)

        test = samples[:test_split]
        validation = samples[test_split : test_split + validation_split]
        print(validation.shape)
        train_pool = samples[test_split + validation_split :]

        if not os.path.isdir(cfg.dir):
            os.makedirs(cfg.dir)

        print("\nSaving generated dataset")
        np.save(os.path.join(cfg.dir, "test"), test, allow_pickle=False)
        np.save(os.path.join(cfg.dir, "val"), validation, allow_pickle=False)
        np.save(os.path.join(cfg.dir, "train_pool"), train_pool, allow_pickle=False)
    else:
        test = np.load(os.path.join(cfg.dir, "test.npy"))
        validation = np.load(os.path.join(cfg.dir, "val.npy"))
        train_pool = np.load(os.path.join(cfg.dir, "train_pool.npy"))

        assert test.shape == (test_split, 50 + 1, 2), (
            f"Loaded test set has shape {test.shape} expected {(test_split, 50 + 1, 2)} according to config"
        )
        assert validation.shape == (validation_split, 50 + 1, 2), (
            f"Loaded validation set has shape {validation.shape} expected {(validation_split, 50 + 1, 2)} according to config"
        )
        assert (
            train_pool.shape[1:] == (50 + 1, 2) and train_pool.shape[0] >= train_split
        ), (
            f"Loaded train set has shape {train_pool.shape} expected at least {(train_split, 50 + 1, 2)} according to config"
        )

    test_dataset = PiVAEDataset(test)
    validation_dataset = PiVAEDataset(validation)
    train_dataset = PiVAEDataset(np.random.permutation(train_pool)[:train_split])

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, collate_fn=np_collate, shuffle=True
    )
    val_loader = DataLoader(
        validation_dataset,
        batch_size=cfg.batch_size,
        collate_fn=np_collate,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.batch_size, collate_fn=np_collate, shuffle=False
    )

    return train_loader, val_loader, test_loader
