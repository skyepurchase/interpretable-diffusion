import os
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.pivae import PiVAEDataset
from datasets.utils import np_collate
from utils.config.datasets import PiVAEConfig


def burgers_PDE(ds: float) -> Callable[[NDArray, NDArray], NDArray]:
    """
    Create the Burgers equation PDE randomly sampling coefficients. Returns a function that can be utilisied by an ODE solver.

    Parameters
    ----------
    ds : float
        The discretisation size

    Returns
    -------
    func : (t, y) -> dy
        The function that progresses y according to the ODE.
    """
    nu = np.random.uniform(low=0.001, high=0.01, size=(1,))

    def step(t: NDArray, y: NDArray) -> NDArray:
        assert len(y.shape) == 1

        # Periodic boundary condition
        looped_y = np.concatenate([y[-1, None], y, y[0, None]], axis=0)

        # Diffusion term
        d2y_ds2 = np.gradient(np.gradient(looped_y, ds), ds)

        # Finite volume flux
        mean_flux = np.square(looped_y) / 2
        flux_interface = np.zeros(len(looped_y) + 1)

        sign = np.zeros_like(flux_interface)
        sign[1:-1] = (looped_y[:-1] + looped_y[1:]) / 2

        flux_interface[1:-1] = np.where(sign[1:-1] > 0, mean_flux[:-1], mean_flux[1:])

        # Viscous Burgers
        looped_dy_dt = (nu * d2y_ds2) + (flux_interface[:-1] - flux_interface[1:]) / ds

        return looped_dy_dt[1:-1]

    return step


def generate_burgers_data(
    n_samples: int,
    t_grid: int = 128,
    x_grid: int = 128,
    T: float = 0.1,
    L: float = 1.0,
    waves: int = 4,
) -> NDArray:
    """
    Using the Burgers equation PDE generate a set of randomly initialised samples.

    Parameters
    ----------
    n_samples : int = 3500
        The number of samples to generate

    Returns
    -------
    samples : (B,T,2)
        B samples with T timesteps including both the timestep and the pendulum angle.
    """
    t_span = (0, T)
    s = np.linspace(0, L, x_grid)
    ds = s[1] - s[0]

    samples = np.empty((0, x_grid + 1, t_grid))
    for _ in tqdm(range(n_samples), desc="Generating samples"):
        y = np.zeros_like(s)
        for _ in range(2):
            A = np.random.rand()
            k = 2 * np.pi * np.random.randint(1, waves) / L
            phi = np.random.uniform(low=0, high=2 * np.pi)
            y += A * np.sin(k * s + phi)

        sol = solve_ivp(
            fun=burgers_PDE(ds),
            t_span=t_span,
            y0=y,
            t_eval=np.linspace(0, T, t_grid),
            method="LSODA",
            rtol=1e-3,
            atol=1e-6,
        )

        series = np.concatenate([sol.t[None], sol.y], axis=0)
        assert series.shape == (x_grid + 1, t_grid), (
            f"Expected series to have shape ({x_grid}, {t_grid}), received {series.shape}"
        )
        samples = np.append(samples, series[None], axis=0)

    return samples


def get_burgers(cfg: PiVAEConfig) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create burgers equation train, val, and test dataloaders"""
    assert cfg.test_split + cfg.validation_split + cfg.train_split <= cfg.n_samples, (
        "Not enough samples specified for desired splits"
    )

    t_grid = 128
    x_grid = 128

    if not all(
        [
            os.path.exists(os.path.join(cfg.dir, cfg.system, file))
            for file in ["test.npy", "val.npy", "train_pool.npy"]
        ]
    ):
        samples = generate_burgers_data(cfg.n_samples, t_grid=t_grid, x_grid=x_grid)
        assert samples.shape == (cfg.n_samples, x_grid + 1, t_grid), (
            f"Expected shape {(cfg.n_samples, x_grid + 1, t_grid)} but received {samples.shape}"
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

        assert test.shape == (cfg.test_split, x_grid + 1, t_grid), (
            f"Loaded test set has shape {test.shape} expected {(cfg.test_split, x_grid + 1, t_grid)} according to config"
        )
        assert validation.shape == (cfg.validation_split, x_grid + 1, t_grid), (
            f"Loaded validation set has shape {validation.shape} expected {(cfg.validation_split, x_grid + 1, t_grid)} according to config"
        )
        assert (
            train_pool.shape[-1] == t_grid and train_pool.shape[0] >= cfg.train_split
        ), (
            f"Loaded train set has shape {train_pool.shape} expected at least {(cfg.train_split, x_grid + 1, t_grid)} according to config"
        )

    test_dataset = PiVAEDataset(test, cfg.retain_dim)
    validation_dataset = PiVAEDataset(validation, cfg.retain_dim)
    train_dataset = PiVAEDataset(
        np.random.permutation(train_pool)[: cfg.train_split], cfg.retain_dim
    )

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
