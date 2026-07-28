from typing import Callable

import jax
import jax.numpy as jnp
from diffrax import Euler


def pendulum_model(y: jax.Array, omega: jax.Array) -> jax.Array:
    """
    Calculate the simple harmonic oscillator ODE.
    d2y_dt2 + omega*sin(y) = 0

    Parameters
    ----------
    y : (B, 2)
        The input dynamics, y and dy
    omega : (1,)
        The known parameter of the pendulum system

    Returns
    -------
    dy : (B, 2)
        The output dynamics, dy and ddy
    """
    return jnp.concatenate(
        [y[:, 1].reshape(-1, 1), -(omega**2) * jnp.sin(y[:, 0].reshape(-1, 1))], axis=1
    )


def make_diffusion_model(
    ds: jax.Array, boundary: str | float = 0.0
) -> Callable[[jax.Array, jax.Array], jax.Array]:
    """Create diffusion physics model with given spatial discretisation"""

    assert boundary == "periodic" or isinstance(boundary, float), (
        f"Expected boundary to be either be 'periodic' or a float, received {boundary}."
    )

    def diffusion_model(y: jax.Array, a: jax.Array) -> jax.Array:
        """
        Calculate the simple diffusion ODE.
        dy_dt - a*d2y_ds2 = 0

        Parameters
        ----------
        y : (B, L)
            The input dynamics, y
        a : (B,)
            The known parameter of the diffusion system

        Returns
        -------
        dy : (B, L)
            The output dynamics, dy_dt
        """
        assert len(y.shape) == 2, (
            f"Expected y to have shape (B, L) but received {y.shape}."
        )
        assert y.shape[0] == a.shape[0], (
            f"Expected y and a to have the same batch, {y.shape[0]} != {a.shape[0]}."
        )

        if boundary == "periodic":
            bound_y = jnp.concatenate([y[:, -2:], y, y[:, :2]], axis=1)
        elif isinstance(boundary, float):
            bound_y = y[1:-1]
        else:
            raise ValueError(
                f"Expected boundary to be either be 'periodic' or a float, received {boundary}."
            )

        # finite differences
        dy_ds = jnp.gradient(bound_y, ds, axis=1)
        assert not isinstance(dy_ds, list)

        d2y_ds2 = jnp.gradient(dy_ds, ds, axis=1)
        assert not isinstance(d2y_ds2, list)

        # generate derivatives
        dy_dt = a * d2y_ds2

        if boundary == "periodic":
            return dy_dt[:, 2:-2]
        elif isinstance(boundary, float):
            return jnp.pad(dy_dt, ((0, 0), (1, 1)), constant_values=boundary)
        else:
            raise ValueError(
                f"Expected boundary to be either be 'periodic' or a float, received {boundary}."
            )

    return diffusion_model


def get_physics_model(
    system: str,
) -> tuple[Callable[[jax.Array, str | float], Callable[..., jax.Array]], int, int]:
    """
    Helper function to get a desired physics system model along with the expected number of physics parameters and time derivatives.

    Parameters
    ----------
    system : str
        The name of the system

    Returns
    model : Callable
        The physics model of the system
    num_parameters : int
        The number of parameters for the model
    num_derivatives : int
        The number of time derivatives of the model
    """
    if system == "pendulum":
        return lambda _ds, _boundary: pendulum_model, 1, 2
    if system == "diffusion":
        return make_diffusion_model, 1, 1

    raise NotImplementedError(f"{system} is not implemented.")


def get_solver(name: str):
    if name == "euler":
        return Euler

    raise NotImplementedError(f"{name} is not implemented.")


def gen_selections():
    pass
