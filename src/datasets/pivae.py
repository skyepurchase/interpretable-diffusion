import os
from typing import Callable

import jax
import jax.numpy as jnp
import torch
from diffrax import Euler, ODETerm, SaveAt, diffeqsolve
from diffrax._custom_types import RealScalarLike
from jaxtyping import Array, PRNGKeyArray, PyTree
from torch.utils.data import DataLoader, Dataset

from datasets.utils import np_collate


class PiVAEDataset(Dataset):
    """A wrapper of the torch Dataset for the PiVAE datasets."""

    def __init__(
        self, series: Array, coefficients: Array, retain_dim: bool = False
    ) -> None:
        super().__init__()
        # series is in the form (B, x, t) flattened would give (B, t*x)
        # Instead (B, x*t) is desired so data[:, :12] gives the initial condition. Thus (B, t, x)
        time_forward = jnp.permute_dims(series, (0, 2, 1))

        if retain_dim:
            self.data = torch.tensor(time_forward)
        else:
            self.data = torch.tensor(time_forward).flatten(start_dim=1)

        self.coefficients = torch.tensor(coefficients)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        """"""
        return self.data[index], self.coefficients[index]


def pendulum_ODE() -> Callable[[RealScalarLike, PyTree, PyTree], PyTree]:
    """
    Create the force damped pendulum from A. Kalousis & N. Takeishi. This randomly samples the constants omega, xi, phi, and A from the ODE and returns a function that given the timestep and state returns the change in state.

    Returns
    -------
    func : (t, y, args) -> dy
        The function that progresses y according to the ODE.
    """

    def step(t: RealScalarLike, y: PyTree, args: PyTree) -> PyTree:
        assert len(y.shape) == 2, (
            f"Expected y to have shape (B, L) but received {y.shape}."
        )
        assert y.shape[0] == args.shape[0], (
            f"Expected y and args to have the same batch size, {y.shape[0]} != {args.shape[0]}"
        )

        omega, xi, phi, A = jnp.split(args, 4, axis=-1)
        theta, theta_dot = jnp.split(y, 2, axis=-1)

        dtheta_dot = (
            (A * omega**2 * jnp.cos(2 * jnp.pi * phi * t))
            - (xi * theta_dot)
            - (omega**2 * jnp.sin(theta))
        )

        return jnp.concatenate([theta_dot, dtheta_dot], axis=-1)

    return step


def generate_pendulum_data(
    n_samples: int = 3500,
    t_grid: int = 50,
    x_grid: int = 1,
    T: float = 2.5,
    *,
    key: PRNGKeyArray,
) -> tuple[Array, Array]:
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
    t = jnp.linspace(0, T, t_grid)
    dt = t[1] - t[0]

    init_key, omega_key, xi_key, phi_key, amp_key, key = jax.random.split(key, num=6)

    y = jax.random.uniform(
        init_key, (n_samples, 2), minval=-jnp.pi / 2, maxval=jnp.pi / 2
    )

    omega = jax.random.uniform(
        omega_key, (n_samples, 1), minval=jnp.pi / 4, maxval=jnp.pi
    )
    xi = jax.random.uniform(xi_key, (n_samples, 1), minval=0, maxval=8)
    phi = jax.random.uniform(phi_key, (n_samples, 1), minval=jnp.pi, maxval=2 * jnp.pi)
    A = jax.random.uniform(amp_key, (n_samples, 1), minval=0, maxval=40)

    args = jnp.concatenate([omega, xi, phi, A], axis=-1)

    ode = pendulum_ODE()

    # sol : (T, B, X)
    sol = diffeqsolve(
        ODETerm(ode), Euler(), t0=0, t1=T, dt0=dt, y0=y, args=args, saveat=SaveAt(ts=t)
    ).ys
    if sol is None:
        raise RuntimeError("ODE solution returned None")

    # samples : (B, 1, T)
    samples = jnp.permute_dims(sol, (1, 2, 0))[:, :1]

    return samples, omega


def diffusion_PDE(ds: Array) -> Callable[[RealScalarLike, PyTree, PyTree], PyTree]:
    """
    Create the diffusion-advection dataset from A. Kalousis & N. Takeishi. TAkes a spatial discretisation and returns a function that given the timestep and state returns the change in state.

    This uses spatial discretisation to calculate the partial derivatives.

    Parameters
    ----------
    ds : float
        The discretisation size

    Returns
    -------
    func : (t, y, args) -> dy
        The function that progresses y according to the ODE.
    """

    def step(_t: RealScalarLike, y: PyTree, args: PyTree) -> PyTree:
        assert len(y.shape) == 2, (
            f"Expected y to have shape (B, L) but received {y.shape}."
        )
        assert y.shape[0] == args.shape[0], (
            f"Expected y and args to have the same batch size, {y.shape[0]} != {args.shape[0]}"
        )

        a, b = jnp.split(args, 2, axis=-1)

        dy_ds = jnp.gradient(y, ds, axis=1)
        assert not isinstance(dy_ds, list)

        d2y_ds2 = jnp.gradient(dy_ds, ds, axis=1)
        assert not isinstance(d2y_ds2, list)

        dy_dt = a * d2y_ds2 - b * dy_ds

        return jnp.pad(dy_dt[:, 1:-1], ((0, 0), (1, 1)), constant_values=0)

    return step


def generate_diffusion_data(
    n_samples: int = 3500,
    t_grid: int = 50,
    x_grid: int = 12,
    T: float = 1.0,
    X: float = 2.0,
    *,
    key: PRNGKeyArray,
) -> tuple[Array, Array]:
    """
    Using the diffusion PDE above generate a set of randomly initialised samples.

    Parameters
    ----------
    n_samples : int = 3500
        The number of samples to generate
    t_grid : int = 50
        The number of timesteps to take

    Returns
    -------
    samples : (B,T,2)
        B samples with T timesteps including both the timestep and the pendulum angle.
    """
    t = jnp.linspace(0, T, t_grid)
    dt = t[1] - t[0]

    s = jnp.linspace(0, X, x_grid)
    ds = s[1] - s[0]

    init_key, a_key, b_key, key = jax.random.split(key, num=4)

    c = jax.random.uniform(init_key, (n_samples, 1), minval=0.5, maxval=1.5)
    y = c * jnp.sin(jnp.pi * s / s.max())

    a = jax.random.uniform(a_key, (n_samples, 1), minval=0.01, maxval=0.1)
    b = jax.random.uniform(b_key, (n_samples, 1), minval=0.01, maxval=0.1)

    args = jnp.concatenate([a, b], axis=-1)

    ode = diffusion_PDE(ds)

    sol = diffeqsolve(
        ODETerm(ode), Euler(), t0=0, t1=T, dt0=dt, y0=y, args=args, saveat=SaveAt(ts=t)
    ).ys

    if sol is None:
        raise RuntimeError("ODE solution returned None")

    samples = jnp.permute_dims(sol, (1, 2, 0))

    return samples, b


SYSTEMS = {
    "pendulum": generate_pendulum_data,
    "diffusion": generate_diffusion_data,
}


def get_pivae(
    dir: str,
    system: str,
    n_samples: int = 3500,
    test_split: int = 500,
    validation_split: int = 1000,
    train_split: int = 1000,
    t_grid: int = 50,
    x_grid: int = 12,
    batch_size: int = 128,
    retain_dim: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create pivae train, val, and test dataloaders"""
    assert test_split + validation_split + train_split <= n_samples

    key = jax.random.key(0)

    if not all(
        [
            os.path.exists(os.path.join(dir, system, file))
            for file in [
                "test.npy",
                "test_coeff.npy",
                "val.npy",
                "val_coeff.npy",
                "train_pool.npy",
                "train_coeff.npy",
            ]
        ]
    ):
        print("\nGenerating samples...")

        sample_key, key = jax.random.split(key)

        samples, coeffs = SYSTEMS[system](
            n_samples, t_grid=t_grid, x_grid=x_grid, key=sample_key
        )
        assert samples.shape == (n_samples, samples.shape[1], 50), (
            f"Expected shape {(n_samples, samples.shape[1], 50)} but received {samples.shape}"
        )

        test = samples[:test_split]
        test_coeff = coeffs[:test_split]
        validation = samples[test_split : test_split + validation_split]
        val_coeff = coeffs[test_split : test_split + validation_split]
        train_pool = samples[test_split + validation_split :]
        train_coeff = coeffs[test_split + validation_split :]

        if not os.path.isdir(os.path.join(dir, system)):
            os.makedirs(os.path.join(dir, system))

        print("\nSaving generated dataset...")

        jnp.save(os.path.join(dir, system, "test"), test, allow_pickle=False)
        jnp.save(
            os.path.join(dir, system, "test_coeff"), test_coeff, allow_pickle=False
        )
        jnp.save(os.path.join(dir, system, "val"), validation, allow_pickle=False)
        jnp.save(os.path.join(dir, system, "val_coeff"), val_coeff, allow_pickle=False)
        jnp.save(
            os.path.join(dir, system, "train_pool"),
            train_pool,
            allow_pickle=False,
        )
        jnp.save(
            os.path.join(dir, system, "train_coeff"), train_coeff, allow_pickle=False
        )
    else:
        test = jnp.load(os.path.join(dir, system, "test.npy"))
        test_coeff = jnp.load(os.path.join(dir, system, "test_coeff.npy"))
        validation = jnp.load(os.path.join(dir, system, "val.npy"))
        val_coeff = jnp.load(os.path.join(dir, system, "val_coeff.npy"))
        train_pool = jnp.load(os.path.join(dir, system, "train_pool.npy"))
        train_coeff = jnp.load(os.path.join(dir, system, "train_coeff.npy"))

        assert test.shape == (test_split, test.shape[1], 50), (
            f"Loaded test set has shape {test.shape} expected {(test_split, test.shape[1], 50)} according to config"
        )
        assert validation.shape == (validation_split, validation.shape[1], 50), (
            f"Loaded validation set has shape {validation.shape} expected {(validation_split, validation.shape[1], 50)} according to config"
        )
        assert train_pool.shape[-1] == 50 and train_pool.shape[0] >= train_split, (
            f"Loaded train set has shape {train_pool.shape} expected at least {(train_split, train_pool.shape[1], 50)} according to config"
        )

    perm_key, key = jax.random.split(key)
    train_perm = jax.random.permutation(perm_key, jnp.arange(train_pool.shape[0]))[
        :train_split
    ]

    test_dataset = PiVAEDataset(test, test_coeff, retain_dim)
    validation_dataset = PiVAEDataset(validation, val_coeff, retain_dim)
    train_dataset = PiVAEDataset(
        train_pool[train_perm], train_coeff[train_perm], retain_dim
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, collate_fn=np_collate, shuffle=True
    )
    val_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        collate_fn=np_collate,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, collate_fn=np_collate, shuffle=False
    )

    return train_loader, val_loader, test_loader
