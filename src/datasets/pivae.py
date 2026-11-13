import os
import tomllib
from typing import Callable

import jax
import jax.numpy as jnp
from diffrax import Dopri8, ODETerm, SaveAt, diffeqsolve
from diffrax._custom_types import RealScalarLike
from jaxtyping import Array, PRNGKeyArray, PyTree

from datasets.loader import ArrayLoader

GENERATION_SEED = 169800


def to_time_forward(series: Array, retain_dim: bool = False) -> Array:
    """
    Convert samples of shape (B, x, t) to (B, t, x), flattened to (B, t*x)
    unless the dimensions are retained.
    """
    time_forward = jnp.permute_dims(series, (0, 2, 1))
    if retain_dim:
        return time_forward
    return time_forward.reshape(time_forward.shape[0], -1)


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
    time: float = 2.5,
    length: float = 0.0,
    phys_param_ranges: tuple = (
        (jnp.pi / 4, jnp.pi),  # omega
        (0.0, 0.8),  # xi
        (jnp.pi, 2 * jnp.pi),  # phi
        (0.0, 40.0),  # A
    ),
    *,
    key: PRNGKeyArray,
) -> tuple[Array, Array]:
    """
    Using the force damped ODE above generate a set of randomly initialised samples.

    Parameters
    ----------
    n_samples : int = 3500
        The number of samples to generate
    t_grid : int = 50
        The number of timesteps
    time : float = 2.5
        The final time of the simulation
    phys_param_ranges : tuple of (min, max) pairs
        The sampling ranges of the physics parameters, in the order
        (omega, xi, phi, A)

    Returns
    -------
    samples : (B, 1, T)
        B samples with T timesteps of the pendulum angle.
    omega : (B, 1)
        The angular velocity of each sample.
    """
    dt = time / t_grid
    t = jnp.arange(t_grid) * dt

    init_key, omega_key, xi_key, phi_key, amp_key, key = jax.random.split(key, num=6)

    y = jnp.concatenate(
        [
            jax.random.uniform(
                init_key, (n_samples, 1), minval=-jnp.pi / 2, maxval=jnp.pi / 2
            ),
            jnp.zeros((n_samples, 1)),
        ],
        axis=-1,
    )

    omega = jax.random.uniform(
        omega_key,
        (n_samples, 1),
        minval=phys_param_ranges[0][0],
        maxval=phys_param_ranges[0][1],
    )
    xi = jax.random.uniform(
        xi_key,
        (n_samples, 1),
        minval=phys_param_ranges[1][0],
        maxval=phys_param_ranges[1][1],
    )
    phi = jax.random.uniform(
        phi_key,
        (n_samples, 1),
        minval=phys_param_ranges[2][0],
        maxval=phys_param_ranges[2][1],
    )
    A = jax.random.uniform(
        amp_key,
        (n_samples, 1),
        minval=phys_param_ranges[3][0],
        maxval=phys_param_ranges[3][1],
    )

    args = jnp.concatenate([omega, xi, phi, A], axis=-1)

    ode = pendulum_ODE()

    # sol : (T, B, X)
    sol = diffeqsolve(
        ODETerm(ode),
        Dopri8(),
        t0=0,
        t1=time,
        dt0=dt,
        y0=y,
        args=args,
        saveat=SaveAt(ts=t),
    ).ys
    if sol is None:
        raise RuntimeError("ODE solution returned None")

    # samples : (B, 1, T)
    return jnp.permute_dims(sol, (1, 2, 0))[:, :1], omega


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

        diff_coeff, adv_coeff = jnp.split(args, 2, axis=-1)

        padded = jnp.pad(y, ((0, 0), (1, 1)), mode="edge")
        dy_ds = (padded[:, 2:] - padded[:, :-2]) / (2 * ds)
        d2y_ds2 = (padded[:, 2:] - 2 * padded[:, 1:-1] + padded[:, :-2]) / (ds**2)

        dy_dt = diff_coeff * d2y_ds2 - adv_coeff * dy_ds

        return jnp.pad(dy_dt[:, 1:-1], ((0, 0), (1, 1)), constant_values=0)

    return step


def generate_diffusion_data(
    n_samples: int = 3500,
    t_grid: int = 50,
    x_grid: int = 12,
    time: float = 1.0,
    length: float = 2.0,
    phys_param_ranges: tuple = (
        (0.01, 0.1),  # diff_coeff
        (0.01, 0.1),  # adv_coeff
    ),
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
    time : float = 1.0
        The final time of the simulation
    length : float = 2.0
        The spatial extent of the domain
    phys_param_ranges : tuple of (min, max) pairs
        The sampling ranges of the physics parameters, in the order
        (diff_coeff, adv_coeff)

    Returns
    -------
    samples : (B, X, T)
        B samples with X spatial grid points and T timesteps, following the
        diffusion-advection PDE with per-sample diff_coeff and adv_coeff.
    diff_coeff : (B, 1)
        The diffusion coefficient of each sample.
    """
    dt = time / t_grid
    t = jnp.arange(t_grid) * dt

    s = jnp.linspace(0, length, x_grid)
    ds = s[1] - s[0]

    init_key, diff_coeff_key, adv_coeff_key, key = jax.random.split(key, num=4)

    c = jax.random.uniform(init_key, (n_samples, 1), minval=0.5, maxval=1.5)
    y = c * jnp.sin(jnp.pi * s / s.max())

    diff_coeff = jax.random.uniform(
        diff_coeff_key,
        (n_samples, 1),
        minval=phys_param_ranges[0][0],
        maxval=phys_param_ranges[0][1],
    )
    adv_coeff = jax.random.uniform(
        adv_coeff_key,
        (n_samples, 1),
        minval=phys_param_ranges[1][0],
        maxval=phys_param_ranges[1][1],
    )

    args = jnp.concatenate([diff_coeff, adv_coeff], axis=-1)

    ode = diffusion_PDE(ds)

    sol = diffeqsolve(
        ODETerm(ode),
        Dopri8(),
        t0=0,
        t1=time,
        dt0=dt,
        y0=y,
        args=args,
        saveat=SaveAt(ts=t),
    ).ys

    if sol is None:
        raise RuntimeError("ODE solution returned None")

    noise_key, key = jax.random.split(key, 2)

    return jnp.permute_dims(sol, (1, 2, 0)) + 1e-3 * jax.random.normal(
        noise_key, (n_samples, x_grid, t_grid)
    ), diff_coeff


def burgers_ODE(ds: Array) -> Callable[[RealScalarLike, PyTree, PyTree], PyTree]:
    """
    Create the Burgers equation PDE randomly sampling coefficients. Returns a function that can be utilisied by an ODE solver.

    Parameters
    ----------
    ds : float
        The discretisation size

    Returns
    -------
    func : (t, y, phys_param) -> dy
        The function that progresses y according to the ODE.
    """

    def step(_t: RealScalarLike, y: PyTree, phys_param: PyTree) -> PyTree:
        assert len(y.shape) == 2, (
            f"Expected y to have shape (B, L) but received {y.shape}."
        )
        assert y.shape[0] == phys_param.shape[0], (
            f"Expected y and phys_param to have the same batch size, {y.shape[0]} != {phys_param.shape[0]}"
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
        looped_dy_dt = (phys_param * d2y_ds2) + (
            flux_interface[:, :-1] - flux_interface[:, 1:]
        ) / ds

        return looped_dy_dt[:, 2:-2]

    return step


def generate_burgers_data(
    n_samples: int,
    t_grid: int = 128,
    x_grid: int = 128,
    time: float = 0.1,
    length: float = 1.0,
    waves: int = 4,
    phys_param_ranges: tuple = (
        (0.001, 0.01),  # viscosity
    ),
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
    time : float = 0.1
        The final time period
    length : float = 1.0
        The length of the space
    waves : int = 4
        The maximum number of waves in the input
    phys_param_ranges : tuple of (min, max) pairs
        The sampling ranges of the physics parameters, in the order (viscosity,)

    Returns
    -------
    samples : (B, X, T)
        B samples with X spatial grid points and T timesteps.
    phys_param : (B, 1)
        The viscosity parameter of each sample.
    """
    dt = time / t_grid
    t = jnp.arange(t_grid) * dt
    s = jnp.linspace(0, length, x_grid, endpoint=False)
    ds = s[1] - s[0]

    y = jnp.zeros((n_samples, x_grid))
    for _ in range(2):
        key, amp_key, freq_key, phase_key = jax.random.split(key, num=4)

        amp = jax.random.uniform(amp_key, (n_samples, 1), minval=0, maxval=1)
        freq = (
            2
            * jnp.pi
            * jax.random.randint(freq_key, (n_samples, 1), minval=1, maxval=waves)
            / length
        )
        phi = jax.random.uniform(phase_key, (n_samples, 1), minval=0, maxval=2 * jnp.pi)

        amp_cast = jnp.tile(amp, reps=(1, x_grid))
        phi_cast = jnp.tile(phi, reps=(1, x_grid))

        y += amp_cast * jnp.sin((freq * s[None, :]) + phi_cast)

    ode = burgers_ODE(ds)

    key, param_key = jax.random.split(key)
    phys_param = jax.random.uniform(
        param_key,
        (n_samples, 1),
        minval=phys_param_ranges[0][0],
        maxval=phys_param_ranges[0][1],
    )

    # sol : (T, B, X)
    sol = diffeqsolve(
        ODETerm(ode),
        Dopri8(),
        t0=0,
        t1=time,
        dt0=dt,
        y0=y,
        args=phys_param,
        saveat=SaveAt(ts=t),
    ).ys
    if sol is None:
        raise RuntimeError("ODE solution returned None")

    # samples : (B, X + 1, T)
    return jnp.permute_dims(sol, (1, 2, 0)), phys_param


SYSTEMS = {
    "pendulum": generate_pendulum_data,
    "diffusion": generate_diffusion_data,
    "burgers": generate_burgers_data,
}


def get_pivae(
    dir: str,
    system: str,
    length: float,
    time: float,
    phys_param_ranges: list | None = None,
    n_samples: int = 3000,
    test_split: int = 1000,
    validation_split: int = 500,
    train_split: int = 1000,
    t_grid: int = 50,
    x_grid: int = 12,
    batch_size: int = 128,
    retain_dim: bool = False,
    *,
    key: PRNGKeyArray,
) -> tuple[ArrayLoader, ArrayLoader, ArrayLoader]:
    """Create pivae train, val, and test dataloaders.

    The dataset is defined by this configuration (domain, grids, parameter
    ranges, and split sizes); the generation seed is fixed so the dataset is
    identical regardless of the run seed.
    """
    assert test_split + validation_split + train_split <= n_samples

    system_dir = os.path.join(dir, system)
    manifest_path = os.path.join(system_dir, "manifest.toml")
    splits = (
        "test",
        "test_coeff",
        "val",
        "val_coeff",
        "train_pool",
        "train_coeff",
    )

    # The manifest holds one [grid_size.<t_grid>] section per dataset
    manifest = {}
    if os.path.exists(manifest_path):
        with open(manifest_path, "rb") as f:
            manifest = tomllib.load(f)
    manifest["grid_size"] = {
        grid_id: grid
        for grid_id, grid in manifest.get("grid_size", {}).items()
        if isinstance(grid, dict)
    }

    entry = manifest["grid_size"].get(str(t_grid), {})
    files = entry.get("files", {})
    cache_valid = (
        bool(entry)
        and entry.get("x_grid") == x_grid
        and entry.get("time") == time
        and all(
            role in files and os.path.exists(os.path.join(system_dir, files[role]))
            for role in splits
        )
    )

    if cache_valid:
        test = jnp.load(os.path.join(system_dir, files["test"]))
        test_coeff = jnp.load(os.path.join(system_dir, files["test_coeff"]))
        validation = jnp.load(os.path.join(system_dir, files["val"]))
        val_coeff = jnp.load(os.path.join(system_dir, files["val_coeff"]))
        train_pool = jnp.load(os.path.join(system_dir, files["train_pool"]))
        train_coeff = jnp.load(os.path.join(system_dir, files["train_coeff"]))

        assert test.shape == (test_split, x_grid, t_grid), (
            f"Loaded test set has shape {test.shape} expected {(test_split, x_grid, t_grid)} according to config"
        )
        assert validation.shape == (validation_split, x_grid, t_grid), (
            f"Loaded validation set has shape {validation.shape} expected {(validation_split, x_grid, t_grid)} according to config"
        )
        assert (
            train_pool.shape[-1] == t_grid
            and train_pool.shape[1] == x_grid
            and train_pool.shape[0] >= train_split
        ), (
            f"Loaded train set has shape {train_pool.shape} expected at least {(train_split, x_grid, t_grid)} according to config"
        )
    else:
        print("\nGenerating samples...")

        sample_key = jax.random.key(GENERATION_SEED)

        gen_kwargs = {}
        if phys_param_ranges is not None:
            gen_kwargs["phys_param_ranges"] = phys_param_ranges

        samples, coeffs = SYSTEMS[system](
            n_samples,
            t_grid=t_grid,
            x_grid=x_grid,
            time=time,
            length=length,
            **gen_kwargs,
            key=sample_key,
        )
        assert samples.shape == (n_samples, x_grid, t_grid), (
            f"Expected shape {(n_samples, x_grid, t_grid)} but received {samples.shape}"
        )

        test = samples[:test_split]
        test_coeff = coeffs[:test_split]
        validation = samples[test_split : test_split + validation_split]
        val_coeff = coeffs[test_split : test_split + validation_split]
        train_pool = samples[test_split + validation_split :]
        train_coeff = coeffs[test_split + validation_split :]

        if not os.path.isdir(system_dir):
            os.makedirs(system_dir)

        print("\nSaving generated dataset...")

        arrays = {
            "test": test,
            "test_coeff": test_coeff,
            "val": validation,
            "val_coeff": val_coeff,
            "train_pool": train_pool,
            "train_coeff": train_coeff,
        }
        for role, array in arrays.items():
            jnp.save(
                os.path.join(system_dir, f"{role}_{t_grid}"),
                array,
                allow_pickle=False,
            )

        manifest["grid_size"][str(t_grid)] = {
            "x_grid": x_grid,
            "time": time,
            "files": {role: f"{role}_{t_grid}.npy" for role in splits},
        }
        lines = []
        for grid_id, grid in manifest["grid_size"].items():
            lines.append(f"[grid_size.{grid_id}]")
            lines.append(f"x_grid = {grid['x_grid']}")
            lines.append(f"time = {grid['time']!r}")
            lines.append("")
            lines.append(f"[grid_size.{grid_id}.files]")
            for role, name in grid["files"].items():
                lines.append(f'{role} = "{name}"')
            lines.append("")
        with open(manifest_path, "w") as f:
            f.write("\n".join(lines).rstrip() + "\n")

    perm_key, key = jax.random.split(key)
    train_perm = jax.random.permutation(perm_key, jnp.arange(train_pool.shape[0]))[
        :train_split
    ]

    train_data = to_time_forward(train_pool[train_perm], retain_dim)
    val_data = to_time_forward(validation, retain_dim)
    test_data = to_time_forward(test, retain_dim)

    train_loader = ArrayLoader(
        train_data, train_coeff[train_perm], batch_size, key=key, shuffle=True
    )
    val_loader = ArrayLoader(val_data, val_coeff, batch_size, key=key, shuffle=False)
    test_loader = ArrayLoader(test_data, test_coeff, batch_size, key=key, shuffle=False)

    return train_loader, val_loader, test_loader
