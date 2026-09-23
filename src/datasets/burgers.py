import os
from typing import Callable

import jax
import jax.numpy as jnp
from diffrax import Euler, ODETerm, SaveAt, diffeqsolve
from diffrax._custom_types import RealScalarLike
from jaxtyping import Array, PRNGKeyArray, PyTree
from torch.utils.data import DataLoader

from datasets.pivae import PiVAEDataset
from datasets.utils import np_collate


def burgers_ODE(ds: Array) -> Callable[[RealScalarLike, PyTree, PyTree], PyTree]:
    """
    Create the Burgers equation PDE randomly sampling coefficients. Returns a function that can be utilisied by an ODE solver.

    Parameters
    ----------
    ds : float
        The discretisation size

    Returns
    -------
    func : (t, y, nu) -> dy
        The function that progresses y according to the ODE.
    """

    def step(_t: RealScalarLike, y: PyTree, nu: PyTree) -> PyTree:
        assert len(y.shape) == 2, (
            f"Expected y to have shape (B, L) but received {y.shape}."
        )
        assert y.shape[0] == nu.shape[0], (
            f"Expected y and nu to have the same batch size, {y.shape[0]} != {nu.shape[0]}"
        )

        # Periodic boundary condition
        looped_y = jnp.concatenate([y[:, -2:], y, y[:, :2]], axis=1)

        # Diffusion term
        dy_ds = jnp.gradient(looped_y, ds, axis=1)
        assert not isinstance(dy_ds, list), (
            f"Expected dy_ds to be Array but found {type(dy_ds)}"
        )
        d2y_ds2 = jnp.gradient(dy_ds, ds, axis=1)
        assert not isinstance(d2y_ds2, list), (
            f"Expected d2y_ds2 to be Array but found {type(d2y_ds2)}"
        )

        # Finite volume flux
        mean_flux = jnp.square(looped_y) / 2
        sign = jnp.pad((looped_y[:, :-1] + looped_y[:, 1:]) / 2, ((0, 0), (1, 1)))
        flux_interface = jnp.pad(
            jnp.where(sign[:, 1:-1] > 0, mean_flux[:, :-1], mean_flux[:, 1:]),
            ((0, 0), (1, 1)),
        )

        # Viscous Burgers
        looped_dy_dt = (nu * d2y_ds2) + (
            flux_interface[:, :-1] - flux_interface[:, 1:]
        ) / ds

        return looped_dy_dt[:, 2:-2]

    return step


def generate_burgers_data(
    n_samples: int,
    t_grid: int = 128,
    x_grid: int = 128,
    T: float = 0.1,
    L: float = 1.0,
    waves: int = 4,
    *,
    key: PRNGKeyArray,
) -> tuple[Array, Array]:
    """
    Using the Burgers equation PDE generate a set of randomly initialised samples.

    Parameters
    ----------
    n_samples : int = 3500
        The number of samples to generate
    t_grid : int = 128
        The temporal grid size
    x_grid : int = 128
        The spatial grid size
    T : float = 0.1
        The final time period
    L : float = 1.0
        The length of the space
    waves : int = 4
        The maximum number of waves in the input

    Returns
    -------
    samples : (B, T, L + 1)
        B samples with T timesteps including both the timestep and the value.
    """
    print("\nGenerating samples...")

    t = jnp.linspace(0, T, t_grid)
    dt = t[1] - t[0]
    s = jnp.linspace(0, L, x_grid, endpoint=False)
    ds = s[1] - s[0]

    y = jnp.zeros((n_samples, x_grid))
    for _ in range(2):
        key, amp_key, freq_key, phase_key = jax.random.split(key, num=4)

        amp = jax.random.uniform(amp_key, (n_samples, 1), minval=0, maxval=1)
        freq = (
            2
            * jnp.pi
            * jax.random.randint(freq_key, (n_samples, 1), minval=1, maxval=waves)
            / L
        )
        phi = jax.random.uniform(phase_key, (n_samples, 1), minval=0, maxval=2 * jnp.pi)

        amp_cast = jnp.tile(amp, reps=(1, x_grid))
        phi_cast = jnp.tile(phi, reps=(1, x_grid))

        y += amp_cast * jnp.sin((freq * s[None, :]) + phi_cast)

    ode = burgers_ODE(ds)

    key, nu_key = jax.random.split(key)
    nu = jax.random.uniform(nu_key, (n_samples, 1), minval=0.001, maxval=0.01)

    # sol : (T, B, X)
    sol = diffeqsolve(
        ODETerm(ode), Euler(), t0=0, t1=T, dt0=dt, y0=y, args=nu, saveat=SaveAt(ts=t)
    ).ys
    if sol is None:
        raise RuntimeError("ODE solution returned None")

    # samples : (B, X + 1, T)
    samples = jnp.concatenate(
        [
            jnp.tile(t.reshape(1, 1, -1), reps=(n_samples, 1, 1)),
            jnp.permute_dims(sol, (1, 2, 0)),
        ],
        axis=1,
    )

    return samples, nu


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
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create burgers equation train, val, and test dataloaders"""
    assert test_split + validation_split + train_split <= n_samples, (
        "Not enough samples specified for desired splits"
    )

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
        key, gen_key = jax.random.split(key)
        samples, coeffs = generate_burgers_data(
            n_samples, t_grid=t_grid, x_grid=x_grid, key=gen_key
        )
        assert samples.shape == (n_samples, x_grid + 1, t_grid), (
            f"Expected shape {(n_samples, x_grid + 1, t_grid)} but received {samples.shape}"
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

        assert test.shape == (test_split, x_grid + 1, t_grid), (
            f"Loaded test set has shape {test.shape} expected {(test_split, x_grid + 1, t_grid)} according to config"
        )
        assert validation.shape == (validation_split, x_grid + 1, t_grid), (
            f"Loaded validation set has shape {validation.shape} expected {(validation_split, x_grid + 1, t_grid)} according to config"
        )
        assert train_pool.shape[-1] == t_grid and train_pool.shape[0] >= train_split, (
            f"Loaded train set has shape {train_pool.shape} expected at least {(train_split, x_grid + 1, t_grid)} according to config"
        )

    key, perm_key = jax.random.split(key)
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
