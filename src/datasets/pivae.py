import os
from typing import Callable

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from datasets.utils import np_collate
from utils.config.datasets import PiVAEConfig

PRECISION = 1000


class PiVAEDataset(Dataset):
    """A wrapper of the torch Dataset for the PiVAE datasets."""

    def __init__(self, series: NDArray) -> None:
        super().__init__()
        # series is in the form (B, x, t) flattened would give (B, t*x)
        # Instead (B, x*t) is desired so data[:, :12] gives the initial condition. Thus (B, t, x)
        time_forward = np.permute_dims(series[:, 1:], (0, 2, 1))
        self.data = torch.tensor(time_forward).flatten(start_dim=1)
        self.timestep = torch.tensor(series[:, 0]).flatten(start_dim=1)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        """"""
        return self.data[index], self.timestep[index]


def pendulum_ODE() -> Callable[[NDArray, NDArray], NDArray]:
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


def generate_pendulum_data(n_samples: int = 3500, steps: int = 50) -> NDArray:
    """
    Using the force damped ODE above generate a set of randomly initialised samples.

    Parameters
    ----------
    n_samples : int = 3500
        The number of samples to generate
    steps : int = 50
        The number of timesteps

    Returns
    -------
    samples : (B,T,2)
        B samples with T timesteps including both the timestep and the pendulum angle.
    """
    T = steps * 0.05
    t_span = (0, T)

    samples = np.empty((0, 2, steps))
    for _ in tqdm(range(n_samples), desc="Generating samples"):
        y_0 = (np.random.randint(0, PRECISION) * (1.57 + 1.57) / PRECISION) - 1.57
        y0 = [y_0, 0]

        sol = solve_ivp(
            fun=pendulum_ODE(),
            t_span=t_span,
            y0=y0,
            t_eval=np.linspace(0, T, steps),
            method="DOP853",
            rtol=1e-3,
            atol=1e-6,
        )

        series = np.concatenate([sol.t[None], np.expand_dims(sol.y[0], axis=0)], axis=0)
        samples = np.append(samples, series[None], axis=0)

    return samples


def diffusion_PDE(ds: float) -> Callable[[NDArray, NDArray], NDArray]:
    """
    Create the diffusion-advection dataset from A. Kalousis & N. Takeishi. TAkes a spatial discretisation and returns a function that given the timestep and state returns the change in state.

    This uses spatial discretisation to calculate the partial derivatives.

    Parameters
    ----------
    ds : float
        The discretisation size

    Returns
    -------
    func : (t, y) -> dy
        The function that progresses y according to the ODE.
    """
    a = (np.random.randint(0, PRECISION) * (1e-1 - 1e-2) / PRECISION) + 1e-2
    b = (np.random.randint(0, PRECISION) * (1e-1 - 1e-2) / PRECISION) + 1e-2

    def diffusion_advection(t: NDArray, y: NDArray) -> NDArray:
        interior = y[1:-1]

        dy_ds = np.gradient(interior, ds)
        d2y_ds2 = np.gradient(dy_ds, ds)

        dy_dt = a * d2y_ds2 - b * dy_ds

        dy_dt_full = np.zeros_like(y)
        dy_dt_full[1:-1] = dy_dt

        return dy_dt_full

    return diffusion_advection


def generate_diffusion_data(n_samples: int = 3500, steps: int = 50) -> NDArray:
    """
    Using the diffusion PDE above generate a set of randomly initialised samples.

    Parameters
    ----------
    n_samples : int = 3500
        The number of samples to generate
    steps : int = 50
        The number of timesteps to take

    Returns
    -------
    samples : (B,T,2)
        B samples with T timesteps including both the timestep and the pendulum angle.
    """
    T = steps * 0.02
    t_span = (0, T)

    H = 12
    s = np.linspace(0, 2.0, H)
    ds = s[1] - s[0]

    samples = np.empty((0, H + 1, steps))
    for _ in tqdm(range(n_samples), desc="Generating samples"):
        c = (np.random.randint(0, PRECISION) * (1.5 - 0.5) / PRECISION) + 0.5
        y0 = c * np.sin(np.pi * s / s.max())

        sol = solve_ivp(
            fun=diffusion_PDE(ds),
            t_span=t_span,
            y0=y0,
            t_eval=np.linspace(0, T, steps),
            method="DOP853",
            rtol=1e-3,
            atol=1e-6,
        )

        series = np.concatenate([sol.t[None], sol.y], axis=0)
        samples = np.append(samples, series[None], axis=0)

    return samples


SYSTEMS = {
    "pendulum": generate_pendulum_data,
    "diffusion": generate_diffusion_data,
}


def get_pivae(cfg: PiVAEConfig) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create pivae train, val, and test dataloaders"""
    assert cfg.test_split + cfg.validation_split + cfg.train_split <= cfg.n_samples

    if not all(
        [
            os.path.exists(os.path.join(cfg.dir, cfg.system, file))
            for file in ["test.npy", "val.npy", "train_pool.npy"]
        ]
    ):
        samples = SYSTEMS[cfg.system](cfg.n_samples)
        assert samples.shape == (cfg.n_samples, samples.shape[1], 50), (
            f"Expected shape {(cfg.n_samples, samples.shape[1], 50)} but received {samples.shape}"
        )

        test = samples[: cfg.test_split]
        validation = samples[cfg.test_split : cfg.test_split + cfg.validation_split]
        train_pool = samples[cfg.test_split + cfg.validation_split :]

        if not os.path.isdir(os.path.join(cfg.dir, cfg.system)):
            os.makedirs(os.path.join(cfg.dir, cfg.system))

        print("\nSaving generated dataset")
        np.save(os.path.join(cfg.dir, cfg.system, "test"), test, allow_pickle=False)
        np.save(
            os.path.join(cfg.dir, cfg.system, "val"), validation, allow_pickle=False
        )
        np.save(
            os.path.join(cfg.dir, cfg.system, "train_pool"),
            train_pool,
            allow_pickle=False,
        )
    else:
        test = np.load(os.path.join(cfg.dir, cfg.system, "test.npy"))
        validation = np.load(os.path.join(cfg.dir, cfg.system, "val.npy"))
        train_pool = np.load(os.path.join(cfg.dir, cfg.system, "train_pool.npy"))

        assert test.shape == (cfg.test_split, test.shape[1], 50), (
            f"Loaded test set has shape {test.shape} expected {(cfg.test_split, test.shape[1], 50)} according to config"
        )
        assert validation.shape == (cfg.validation_split, validation.shape[1], 50), (
            f"Loaded validation set has shape {validation.shape} expected {(cfg.validation_split, validation.shape[1], 50)} according to config"
        )
        assert train_pool.shape[-1] == 50 and train_pool.shape[0] >= cfg.train_split, (
            f"Loaded train set has shape {train_pool.shape} expected at least {(cfg.train_split, train_pool.shape[1], 50)} according to config"
        )

    test_dataset = PiVAEDataset(test)
    validation_dataset = PiVAEDataset(validation)
    train_dataset = PiVAEDataset(np.random.permutation(train_pool)[: cfg.train_split])

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
