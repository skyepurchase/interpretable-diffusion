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


def make_diffusion_model(ds: jax.Array) -> Callable[[jax.Array, jax.Array], jax.Array]:
    """Create diffusion physics model with given spatial discretisation"""

    def diffusion_model(y: jax.Array, a: jax.Array) -> jax.Array:
        """
        Calculate the simple diffusion ODE.
        dy_dt - a*d2y_ds2 = 0

        Parameters
        ----------
        y : (B, L)
            The input dynamics, y
        a : (1,)
            The known parameter of the diffusion system

        Returns
        -------
        dy : (B, L)
            The output dynamics, dy_dt
        """
        interior = y[:, 1:-1]

        # finite differences
        dy_ds = jnp.gradient(interior, ds, axis=1)
        assert not isinstance(dy_ds, list)

        d2y_ds2 = jnp.gradient(dy_ds, ds, axis=1)
        assert not isinstance(d2y_ds2, list)

        # generate derivatives
        dy_dt = a * d2y_ds2

        full_dy_dt = jnp.pad(dy_dt, ((0, 0), (1, 1)))

        return full_dy_dt

    return diffusion_model


def get_physics_model(
    system: str,
) -> tuple[Callable[[jax.Array], Callable[..., jax.Array]], int, int]:
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
        return lambda _: pendulum_model, 1, 2
    if system == "diffusion":
        return make_diffusion_model, 1, 1

    raise NotImplementedError(f"{system} is not implemented.")


def get_solver(name: str):
    if name == "euler":
        return Euler

    raise NotImplementedError(f"{name} is not implemented.")


def gen_selections():
    pass
