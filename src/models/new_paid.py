import jax
import jax.numpy as jnp
from diffrax import Euler, ODETerm, SaveAt, diffeqsolve
from flax import nnx
from jaxtyping import Array

# from diffuse.neural_network.nn import CondUNet
from models.mlp import Mlp
from models.modules.acts import ACTS
from models.physics_models import Operator, get_physics_model, get_solver


class FlowModel(nnx.Module):
    def __init__(self, timesteps: int = 100):
        t = jnp.linspace(0, 1, timesteps)
        self.dt = t[1] - t[0]

    def step(self, z_t: Array, t: Array) -> Array:
        """
        Single step of the flow model ODE.

        Parameters
        ----------
        z_t : Array (B, W, H, C)
            The current value of the sample
        t : Array (B, 1)
            The current timestep

        Returns
        -------
        dz_dt : Array (B, W, H, C)
            The predicted velocity of z
        """
        raise NotImplementedError

    def __call__(self, s: Array, dt: Array | None = None) -> Array:
        """
        Map the input according to the learnt flow.

        Parameters
        ----------
        s : Array (B, H, W, C)
            The input (source)
        dt : Array ()
            ODE stepsize

        Returns
        -------
        t : Array (B, H, W, C)
            The target according to the learnt flow
        """
        dt = dt if dt is not None else self.dt
        assert dt.dtype == jnp.float32, (
            f"Type of dt should be float, received {dt.dtype}."
        )
        assert dt.shape == (), f"dt should be shapeless, received shape {dt}."

        # s : (B, k, X)
        assert len(s.shape) == 3, (
            f"Expected s to have shape (B, k, X), received {s.shape}."
        )
        batch_size = s.shape[0]
        shape = s.shape

        def ode(t, y, _):
            # y : (B, X*k)
            assert len(y.shape) == 2, (
                f"Expected y to have shape (B, L) but received {y.shape}."
            )

            tau = t * jnp.ones((y.shape[0], 1))
            return self.step(y.reshape(*shape), tau).reshape(batch_size, -1)

        mapped = diffeqsolve(
            ODETerm(ode),
            Euler(),
            t0=0.0,
            t1=1.0,
            dt0=dt,
            y0=s.reshape(batch_size, -1),
            saveat=SaveAt(ts=jnp.ones((1))),
        ).ys

        if mapped is None:
            raise RuntimeError("Failed to solve flow model ODE.")

        return mapped[0]


class FlowModel2D(FlowModel):
    def __init__(
        self,
        in_channels: int,
        base_channel: int,
        ch_mult: tuple[int],
        activation: str,
        dropout: bool,
        dropout_rate: float,
        timesteps: int = 100,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        super().__init__(timesteps)

        self.act = ACTS[activation]

        # self.unet = CondUNet(
        #     in_channels=in_channels,
        #     ch=base_channel,
        #     ch_mult=ch_mult,
        #     num_res_blocks=1,
        #     attention_resolutions=(),
        #     activation=self.act,
        #     dropout=dropout,
        #     dropout_rate=dropout_rate,
        #     num_heads=0,
        #     rngs=rngs,
        # )

    def step(self, z_t: Array, t: Array) -> Array:
        """
        Single step of the flow model ODE.

        Parameters
        ----------
        z_t : Array (B, W, H, C)
            The current value of the sample
        t : Array (B, 1)
            The current timestep

        Returns
        -------
        dz_dt : Array (B, W, H, C)
            The predicted velocity of z
        """
        assert z_t.shape[0] == t.shape[0], (
            f"z_t and t must have the same batch, {z_t.shape[0]} != {t.shape[0]}."
        )

        return z_t  # self.unet(z_t, t).output


class FlowModel1D(FlowModel):
    def __init__(
        self,
        in_features: int,
        hid_features: list[int],
        act: str,
        timesteps: int = 100,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        super().__init__(timesteps)

        self.mlp = Mlp(in_features + 1, hid_features, in_features, act, rngs)

    def step(self, z_t: Array, t: Array) -> Array:
        """
        Single step of the flow model ODE.

        Parameters
        ----------
        z_t : Array (B, H)
            The current value of the sample
        t : Array (B, 1)
            The current timestep

        Returns
        -------
        dz_dt : Array (B, H)
            The predicted velocity of z
        """
        assert z_t.shape[0] == t.shape[0], (
            f"z_t and t must have the same batch, {z_t.shape[0]} != {t.shape[0]}."
        )
        assert t.shape[1:] == (1,), (
            f"Expected t to have shape (B, 1), received {t.shape}."
        )
        assert len(z_t.shape) == 3, (
            f"Expected z_t to have shape (B, k, X), received {z_t.shape}."
        )

        input_shape = z_t.shape

        pred = self.mlp(jnp.concatenate([z_t.reshape(z_t.shape[0], -1), t], axis=1))

        return pred.reshape(*input_shape)


class NewPaid(nnx.Module):
    """
    New physics aligned interpretable diffusion (nuPAID) model.
    """

    def __init__(
        self,
        augmenter_cfg: dict,
        mapper_cfg: dict,
        estimator_cfg: dict,
        operator: str,
        solver: str,
        observed_period: int,
        x_grid: int,
        t_grid: int,
        period: float,
        length: float,
        coeff_ranges: list[tuple[float, float]],
        phys_reg: float,
        aug_reg: float,
        rngs: nnx.Rngs,
    ) -> None:
        super().__init__()
        # Add dataset specific values
        augmenter_cfg["in_features"] = x_grid + 1 + (observed_period * x_grid) + 1
        augmenter_cfg["out_features"] = x_grid

        mapper_cfg["in_features"] = observed_period * x_grid

        estimator_cfg["in_features"] = observed_period * x_grid
        estimator_cfg["out_features"] = 1

        self.augmenter = Mlp(**augmenter_cfg, rngs=rngs)
        self.mapper = FlowModel1D(**mapper_cfg, rngs=rngs)
        self.estimator = Mlp(**estimator_cfg, rngs=rngs)

        self.k = observed_period
        self.x_grid = x_grid
        self.t_grid = t_grid
        self.T = period
        self.t = jnp.linspace(0, period, t_grid)
        self.dt = self.t[1] - self.t[0]
        s = jnp.linspace(0, length, x_grid)
        ds = s[1] - s[0]

        make_operator, _, _ = get_physics_model(operator)
        self.operator = make_operator(ds, "periodic")
        self.solver = get_solver(solver)

        self.coeff_ranges = coeff_ranges

        self.phys_reg = phys_reg
        self.aug_reg = aug_reg

    def _get_augmented_ode(self) -> Operator:
        """Augment the known operator with the augmentation function."""

        def ode(t, y, args):
            assert isinstance(args, Array)
            assert len(args.shape) == 2
            assert args.shape[0] == y.shape[0]
            assert args.shape[1] == 1 + (self.x_grid * self.k)

            dy_dt = self.operator(t, y, [args[:, 0, None]])
            dy_dt += self.augmenter(
                jnp.concatenate([y, args, jnp.ones((y.shape[0], 1))], axis=1)
            )

            return dy_dt

        return ode

    def gen_physics(self, init_y: Array, *coeffs) -> Array:
        """
        Given an initial condition and coefficients generate solutions to the known operator

        Parameters
        ----------
        init_y: Array (B, X)
            The initial condition
        *coeffs
            The coefficients for the known operator

        Returns
        -------
        u : Array (B, T, X)
            The estimated solution
        """
        u = diffeqsolve(
            ODETerm(self.operator),
            self.solver(),
            t0=0.0,
            t1=self.T,
            dt0=self.dt,
            y0=init_y,
            args=coeffs,
            saveat=SaveAt(ts=self.t),
        ).ys

        if u is None:
            raise RuntimeError("Failed to solve known operator.")

        return jnp.permute_dims(u, (1, 0, 2))

    def estimate(self, u: Array) -> Array:
        """
        Given an observed solution to F(u) + L(u) = 0 provide an estimate for the coefficients of L the known operator.

        Parameters
        ----------
        u : Array (B, k, X)
            Observed solution

        Returns
        -------
        coeffs : Array (B, Z)
            Estimated coefficients for L
        """
        assert u.shape[1:] == (self.k, self.x_grid), (
            f"Expected input to be an observed sequence, {u.shape[1:]} != ({self.k}, {self.x_grid})."
        )

        return self.estimator(self.mapper(u))

    def solve(self, u: Array) -> Array:
        """
        Given an observed solution to F(u) + L(u) = 0 forecast the solution until timestep T.

        Parameters
        ----------
        u : Array (B, k, X)
            Observed solution

        Returns
        -------
        u_hat : Array (B, T, X)
            Estimated solution
        """
        assert u.shape[1:] == (self.k, self.x_grid), (
            f"Expected input to be an observed sequence, {u.shape[1:]} != ({self.k}, {self.x_grid})"
        )

        args = jnp.concatenate(
            [self.estimate(u), u.reshape(-1, self.x_grid * self.k)],
            axis=1,
        )
        assert args.shape[0] == u.shape[0], (
            f"Expected args and u to have same batch dimension, {u.shape[0]} != {args.shape[0]}"
        )

        forecast = diffeqsolve(
            ODETerm(self._get_augmented_ode()),
            self.solver(),
            t0=0,
            t1=self.T,
            dt0=self.dt,
            y0=u[:, 0],
            args=jax.lax.stop_gradient(args),
            saveat=SaveAt(ts=self.t),
        ).ys

        if forecast is None:
            raise RuntimeError("Failed to solve augmented ODE.")

        return jnp.permute_dims(forecast, (1, 0, 2))

    def __call__(self, u: Array) -> Array:
        """Wrapper for solve"""
        return self.solve(u)
