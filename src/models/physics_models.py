from typing import Callable

import jax
import jax.numpy as jnp
from diffrax import Euler
from diffrax._custom_types import RealScalarLike
from jaxtyping import PyTree

Operator = Callable[[RealScalarLike, PyTree, PyTree], PyTree]


def pendulum_model(_t: RealScalarLike, y: jax.Array, phys_params: jax.Array) -> jax.Array:
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
    omega = phys_params[0]

    theta, theta_dot = jnp.split(y, 2, axis=-1)
    dtheta_dot = -(omega**2 * jnp.sin(theta))

    return jnp.concatenate([theta_dot, dtheta_dot], axis=1)


def make_diffusion_model(ds: jax.Array, boundary: str | float = 0.0) -> Operator:
    """Create diffusion physics model with given spatial discretisation"""

    assert boundary == "periodic" or isinstance(boundary, float), (
        f"Expected boundary to be either be 'periodic' or a float, received {boundary}."
    )

    def diffusion_model(
        _t: RealScalarLike, y: jax.Array, phys_params: jax.Array
    ) -> jax.Array:
        """
        Calculate the simple diffusion ODE.
        dy_dt - diff_coeff*d2y_ds2 = 0

        Parameters
        ----------
        t : (B,)
            The current timestep
        y : (B, L)
            The input dynamics, y
        phys_params : list[Array]
            The physics parameters of the diffusion system; phys_params[0] is the
            diffusion coefficient diff_coeff

        Returns
        -------
        dy : (B, L)
            The output dynamics, dy_dt
        """
        diff_coeff = phys_params[0]
        assert len(y.shape) == 2, (
            f"Expected y to have shape (B, L) but received {y.shape}."
        )
        assert y.shape[0] == diff_coeff.shape[0], (
            f"Expected y and diff_coeff to have the same batch, {y.shape[0]} != {diff_coeff.shape[0]}."
        )

        if boundary == "periodic":
            bound_y = jnp.concatenate([y[:, -2:], y, y[:, :2]], axis=1)
        else:
            bound_y = y

        # finite differences
        dy_ds = jnp.gradient(bound_y, ds, axis=1)
        assert not isinstance(dy_ds, list)

        d2y_ds2 = jnp.gradient(dy_ds, ds, axis=1)
        assert not isinstance(d2y_ds2, list)

        # generate derivatives
        dy_dt = diff_coeff * d2y_ds2

        if boundary == "periodic":
            return dy_dt[:, 2:-2]
        elif isinstance(boundary, float):
            return jnp.pad(dy_dt[:, 1:-1], ((0, 0), (1, 1)), constant_values=boundary)
        else:
            raise ValueError(
                f"Expected boundary to be either be 'periodic' or a float, received {boundary}."
            )

    return diffusion_model


def get_physics_model(
    system: str,
) -> tuple[Callable[[jax.Array, str | float], Operator], int, int]:
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


def get_solver(name: str) -> type[Euler]:
    if name == "euler":
        return Euler

    raise NotImplementedError(f"{name} is not implemented.")


def gen_selections():
    pass
