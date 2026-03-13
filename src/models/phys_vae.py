from collections.abc import Callable
from functools import reduce

import jax
import jax.numpy as jnp
from diffrax import Euler, ODETerm, SaveAt, diffeqsolve
from diffrax._custom_types import RealScalarLike
from flax import nnx
from jaxtyping import PyTree

from models.mlp import make_mlp
from utils.config.models import PhysDecoderConfig, PhysEncoderConfig, PhysVAEConfig
from utils.misc import Distribution


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


PHYS_MODELS: dict[str, Callable[[jax.Array], Callable[..., jax.Array]]] = {
    "pendulum": lambda _: pendulum_model,
    "diffusion": make_diffusion_model,
}
NUM_PHYS_PARAMS = {
    "pendulum": 1,
    "diffusion": 1,
}
NUM_DERIVATIVES = {
    "pendulum": 2,
    "diffusion": 1,
}
SOLVERS = {"euler": Euler}


def gen_selections(n_elms: int) -> jax.Array:
    """Given a number of elements generate binary representations for selection"""
    combs = 2**n_elms
    numbers = jnp.arange(combs, dtype=jnp.uint8)
    bits = jnp.unpackbits(numbers[:, None], axis=1, count=n_elms, bitorder="little")

    return bits.T.astype(jnp.int32)


def gen_mask(total_combs: int) -> jax.Array:
    """Generate mask for the discrepancy loss"""
    nums = jnp.arange(total_combs)

    count_mask = jax.vmap(
        lambda x: jax.vmap(
            # Get flipped bits and count how many there are
            lambda y: jax.lax.population_count(x ^ y)
        )(nums[1:])
    )(nums[:-1])

    tri_mask = jnp.triu(count_mask)

    return (tri_mask == 1).astype(jnp.uint8)


class PhysEncoder(nnx.Module):
    """
    Physics integrated VAE encoder network.

    Parameters
    ----------
    cfg : PhysEncoderConfig
        The config of the encoder
    time_dim : int
        The time dimension, equivalently the number of timesteps
    rngs : nnx.Rngs
        The model RNGs
    """

    def __init__(self, cfg: PhysEncoderConfig, data_dim: int, rngs: nnx.Rngs) -> None:
        self.data_dim = data_dim

        # Setup auxiliary functions
        self.aux_latent_dims = []
        self.aux_funcs = nnx.List()
        for aux_cfg in cfg.aux_funcs:
            self.aux_latent_dims.append(aux_cfg.latent_dim)

            # Assumes all models are MLPs
            aux_func = make_mlp(
                data_dim, aux_cfg.dims, 2 * aux_cfg.latent_dim, aux_cfg.act, rngs=rngs
            )

            self.aux_funcs.append(aux_func)

        latent_dims = reduce(lambda x, y: x + y, self.aux_latent_dims)

        self.unmixer = make_mlp(
            latent_dims + data_dim,
            cfg.unmixer.dims,
            data_dim,
            cfg.unmixer.act,
            rngs=rngs,
        )
        self.residual = cfg.unmixer.residual

        self.phys_funcs = nnx.List()
        for phys_cfg in cfg.phys_funcs:
            # Assumes all models are MLPs
            phys_func = make_mlp(
                data_dim,
                phys_cfg.dims,
                2 * phys_cfg.latent_dim,
                phys_cfg.act,
                rngs=rngs,
            )

            self.phys_funcs.append(phys_func)

    def __call__(
        self, x: jax.Array
    ) -> tuple[list[Distribution], list[Distribution], jax.Array]:
        """
                Encode a given datapoint into auxiliary and physical latents. Additionally, the estimated physics only signal is returned
        (zero to IDE)
                Parameters
                ----------
                x : Array (B, T)
                    The batched datapoint

                Returns
                -------
                aux_stats : list[Distribution] [(B, Z*2)]
                    The list of auxiliary latent statistics
                aux_stats : list[Distribution] [(B, Z*2)]
                    The list of physical latent statistics
                phys : Array (B, T)
                    The estimated physics only signal
        """
        # Compute Auxiliary latent statistics
        aux_stats = []
        for aux_func in self.aux_funcs:
            mu, log_sigma = jnp.split(aux_func(x), 2, axis=1)
            aux_stats.append((mu, log_sigma))

        res = x if self.residual else 0
        phys = (
            self.unmixer(
                jnp.concatenate([x] + [latent[0] for latent in aux_stats], axis=1)
            )
            + res
        )

        phys_stats = []
        for phys_func in self.phys_funcs:
            mu, log_sigma = jnp.split(phys_func(x), 2, axis=1)
            phys_stats.append((mu, log_sigma))

        return phys_stats, aux_stats, phys


class PhysDecoder(nnx.Module):
    """
    Physics integrated VAE encoder network.

    Parameters
    ----------
    cfg : PhysEncoderConfig
        The config of the encoder
    system : str
        The underlying physical system
    time_dim : int
        The time dimension, equivalently the number of timesteps
    t1 : int
        The final timestep of the simulation
    dt : float
        The length of the timestep
    L : float
        The length of the system
    space_step : int
        The number of spatial discretisations
    integral_level : int
        The solver substeps
    t_span : Array (T)
        The steps to calculate the ODE on
    ode_solver : str
        The ODE solver name
    rngs : nnx.Rngs
        The model RNGs
    """

    def __init__(
        self,
        cfg: PhysDecoderConfig,
        system: str,
        t1: int,
        dt: float,
        L: float,
        time_dim: int,
        num_grid_points: int,
        integral_level: int,
        t_span: jax.Array,
        ode_solver: str,
        rngs: nnx.Rngs,
    ) -> None:
        self.time_dim = time_dim
        self.t1 = t1
        self.dt = dt
        self.integral_level = integral_level
        self.t_span = t_span
        self.num_grid_points = num_grid_points
        self.num_derivatives = NUM_DERIVATIVES.get(system, 1)

        self.data_dim = self.time_dim * self.num_grid_points
        self.ode_dim = self.num_grid_points * self.num_derivatives

        self.x_log_sigma = cfg.x_log_sigma

        self.num_pre_funcs = len(cfg.pre_funcs)
        self.num_post_funcs = len(cfg.post_funcs)

        self.pre_funcs = nnx.List()
        self.post_funcs = nnx.List()
        for func_cfg in cfg.pre_funcs:
            # Assume all models are MLPs
            # TODO: use a time embedding
            # z + L*D + t -> dy
            func = make_mlp(
                func_cfg.latent_dim + self.ode_dim + 1,
                func_cfg.dims,
                self.num_grid_points,  # TODO: this may need to be L*D
                func_cfg.act,
                rngs=rngs,
            )
            self.pre_funcs.append(func)
        self.pre_combs = 2**self.num_pre_funcs
        self.pre_selections = gen_selections(self.num_pre_funcs)

        for func_cfg in cfg.post_funcs:
            func = make_mlp(
                func_cfg.latent_dim + self.data_dim,
                func_cfg.dims,
                self.data_dim - self.num_grid_points,  # Do not overwrite init condition
                func_cfg.act,
                rngs=rngs,
            )
            self.post_funcs.append(func)
        self.post_combs = 2**self.num_post_funcs
        self.post_selections = gen_selections(self.num_pre_funcs + self.num_post_funcs)[
            self.num_pre_funcs :
        ]

        if self.num_grid_points > 1:
            ds = L / (self.num_grid_points - 1)
        else:
            ds = 0.0

        self.physics_model = PHYS_MODELS.get(system, lambda _: None)(
            jnp.array(ds, dtype=jnp.float32)
        )
        if self.physics_model is None:
            raise NotImplementedError(f"Physics model {system} is not implemented.")

        self.solver = SOLVERS.get(ode_solver, None)
        if self.solver is None:
            raise NotImplementedError(f"ODE solver {system} is not implemented.")

        self.mask = gen_mask(self.pre_combs * self.post_combs)

    def full_forward(
        self, phys_params: list[jax.Array], latents: list[jax.Array], init_y: jax.Array
    ) -> Distribution:
        """
        Used during training to produce all combination of auxiliary functions applied the input.

        Parameters
        ----------
        phys_params : list[jax.Array] [(B, Z)]
            The physical latents
        latents : list[jax.Array] [(B, Z)]
            The auxiliary latents
        init_y : Array (B, L)
            The initial condition

        Returns
        -------
        series : Distribution (m, B, T)
            The statistics for the series with the mean and log variance stacked in the first axis
        """

        def ODE(t: RealScalarLike, y: PyTree, _args: PyTree):
            """Internal ODE to be solved during the generation process."""
            # y : (B*n, L*D)
            assert len(y.shape) == 2, f"expected (B*n, L*D) but received {y.shape}"
            assert y.shape[1] == self.ode_dim, (
                f"expected (B*n, L*D) but received {y.shape}"
            )
            # ys : (n, B, L*D)
            ys = jnp.reshape(y, (self.pre_combs, -1, self.ode_dim))

            # ys_dot_phys : (n, B, L*D)
            ys_dot_phys = jax.vmap(self.physics_model, in_axes=(0, None), out_axes=0)(
                ys, *phys_params
            )

            # ys_dot_aux : (n, B, L*D)
            ys_dot_aux = jnp.zeros_like(ys_dot_phys)

            # selections : (n, 1), z_aux : (B, Z)
            for z_aux, pre_func, selections in zip(
                latents[: len(self.pre_funcs)], self.pre_funcs, self.pre_selections
            ):
                # y : (B, 2), apply : (1,)
                def apply_func(y, apply):
                    return apply * jnp.pad(
                        pre_func(
                            jnp.concatenate(
                                [z_aux, y, jnp.ones_like(y)[:, 0, None] * t],
                                axis=1,
                            )
                        ),
                        # TODO: this allows only control over a single derivative. Is this best?
                        ((0, 0), (self.ode_dim - self.num_grid_points, 0)),
                    )

                # ys_dot_aux : (n, B, L*D)
                ys_dot_aux += jax.vmap(apply_func, in_axes=(0, 0), out_axes=0)(
                    ys, selections
                )

            # ys_dot : (B*n, L*D)
            return jnp.reshape(ys_dot_phys, (-1, self.ode_dim))

        # (B, L*D) => (B*n, L*D)
        init_cond = jnp.tile(
            # (B, L) => (B, L*D)
            jnp.pad(
                init_y,
                ((0, 0), (0, self.ode_dim - self.num_grid_points)),
                mode="empty",
            ),
            reps=(self.pre_combs, 1),
        )

        ODE(self.dt, init_cond, 0)

        y_seq = diffeqsolve(
            ODETerm(ODE),
            self.solver(),
            t0=0.0,
            t1=self.t1,
            dt0=self.dt,
            y0=init_cond,
            saveat=SaveAt(ts=self.t_span),
        ).ys
        # y_seq : (S, B*n, L*D)
        assert y_seq is not None, "Failed to solve ODE"

        # trim_y_seq : (T, B*n, L*D)
        trim_y_seq = y_seq[jnp.arange(0, self.t1, self.integral_level)]

        # trim_ys : (T, n, B, L*D)
        trim_ys = jnp.reshape(
            trim_y_seq, (self.time_dim, self.pre_combs, -1, self.ode_dim)
        )
        # trim_y : (n, B, T, L)
        trim_y = jnp.permute_dims(
            trim_ys[:, :, :, : self.num_grid_points], (1, 2, 0, 3)
        )
        # x_mean : (n, B, L*T)
        x_mean = jnp.reshape(trim_y, (self.pre_combs, -1, self.data_dim))
        # x_means : (m, B, L*T)
        x_means = jnp.tile(x_mean, reps=(self.post_combs, 1, 1))

        for z_aux, post_func, selections in zip(
            latents[len(self.pre_funcs) : len(self.post_funcs) + 1],
            self.post_funcs,
            self.post_selections,
        ):
            # x_mean : (B, L*T), apply : (1,)
            def apply_func(x_mean, apply):
                return (1 - apply) * x_mean + apply * jnp.pad(
                    post_func(jnp.concatenate([z_aux, x_mean], axis=1)),
                    ((0, 0), (self.num_grid_points, 0)),
                    mode="empty",
                )

            # x_means : (m, B, L*T)
            x_means += jax.vmap(apply_func, in_axes=(0, 0), out_axes=0)(
                x_means, selections
            )

        return (x_means, jnp.ones_like(x_means) * self.x_log_sigma)

    def __call__(
        self,
        phys_params: list[jax.Array],
        latents: list[jax.Array],
        init_y: jax.Array,
    ) -> Distribution:
        """
        Given physical and auxiliary latents decode the full series based on the initial condition.

        Parameters
        ----------
        phys_params : list[jax.Array] [(B, Z)]
            The physical latents
        latents : list[jax.Array] [(B, Z)]
            The auxiliary latents
        init_y : Array (B, L)
            The initial condition

        Returns
        -------
        series : Distribution (B, T)
            The statistics for the series with the mean and log variance stacked in the first axis
        """

        def ODE(t: RealScalarLike, y: PyTree, _args: PyTree):
            """Internal ODE to be solved during the generation process."""
            # y : (B, L*D)
            assert len(y.shape) == 2, f"expected (B, L*D) but received {y.shape}"
            assert y.shape[1] == self.ode_dim, (
                f"expected (B, L*D) but received {y.shape}"
            )
            # ys : (B, L*D)
            ys = y[:, : self.ode_dim].reshape(-1, self.ode_dim)

            # ys_dot_phys : (B, L*D)
            ys_dot_phys = self.physics_model(ys, *phys_params)

            # ys_dot_aux : (B, L*D)
            ys_dot_aux = jnp.zeros_like(ys_dot_phys)

            # selections : (n, 1), z_aux : (B, Z)
            for z_aux, pre_func, selections in zip(
                latents[: len(self.pre_funcs)], self.pre_funcs, self.pre_selections
            ):
                ys_dot_aux += jnp.pad(
                    pre_func(
                        jnp.concatenate(
                            [z_aux, y, jnp.ones_like(y)[:, 0, None] * t],
                            axis=1,
                        )
                    ),
                    # TODO: this allows only control over a single derivative. Is this best?
                    ((0, 0), (self.ode_dim - self.num_grid_points, 0)),
                )

            # ys_dot : (B, L*D)
            return jnp.reshape(ys_dot_phys + ys_dot_aux, (-1, self.ode_dim))

        # init_cond : (B, L*D)
        init_cond = jnp.pad(init_y, ((0, 0), (0, self.ode_dim - self.num_grid_points)))

        y_seq = diffeqsolve(
            ODETerm(ODE),
            self.solver(),
            t0=0.0,
            t1=self.t1,
            dt0=self.dt,
            y0=init_cond,
            saveat=SaveAt(ts=self.t_span),
        ).ys
        # y_seq : (S, B, L*D)
        assert y_seq is not None, "Failed to solve ODE"

        # trim_y_seq : (T, B, L*D)
        trim_y_seq = y_seq[jnp.arange(0, self.t1, self.integral_level)]

        # x_mean : (B, T, L)
        x_mean = jnp.permute_dims(trim_y_seq[:, :, : self.num_grid_points], (1, 0, 2))
        # x_means : (B, L*T)
        x_means = jnp.reshape(x_mean, (-1, self.data_dim))

        for z_aux, post_func in zip(
            latents[len(self.pre_funcs) : len(self.post_funcs) + 1],
            self.post_funcs,
        ):
            # x_means : (B, L*T)
            x_means += jnp.pad(
                post_func(jnp.concatenate([z_aux, x_means], axis=1)),
                ((0, 0), (self.num_grid_points, 0)),
                mode="empty",
            )

        return (x_means, jnp.ones_like(x_means) * self.x_log_sigma)


class PhysVAE(nnx.Module):
    """
    Physics integrated VAE.

    Parameters
    ----------
    cfg : PhysVAEConfig
        The config for the network including the encoder and decoder
    rngs : nnx.Rngs
        The model RNGs
    """

    def __init__(self, cfg: PhysVAEConfig, rngs: nnx.Rngs) -> None:
        self.integral_level = cfg.integral_level
        self.integral_len = (cfg.time_dim - 1) * self.integral_level + 1
        self.integral_step = cfg.timestep / float(cfg.integral_level)

        self.t_span = jnp.linspace(
            0.0, self.integral_step * (self.integral_len - 1), self.integral_len
        )
        self.num_grid_points = cfg.num_grid_points
        self.ode_dim = self.num_grid_points * NUM_DERIVATIVES.get(cfg.system, 1)
        self.data_dim = self.num_grid_points * cfg.time_dim

        num_phys_params = NUM_PHYS_PARAMS[cfg.system]

        assert len(range(0, self.integral_len, self.integral_level)) == cfg.time_dim, (
            f"Time dimension and integral steps not compatible {len(range(0, self.integral_len, self.integral_level))} != {cfg.time_dim}"
        )

        self.encode = PhysEncoder(
            cfg=cfg.encoder, data_dim=cfg.time_dim * cfg.num_grid_points, rngs=rngs
        )
        self.decode = PhysDecoder(
            cfg=cfg.decoder,
            t1=self.integral_len,
            dt=self.integral_step,
            L=cfg.length,
            time_dim=cfg.time_dim,
            num_grid_points=cfg.num_grid_points,
            integral_level=cfg.integral_level,
            t_span=self.t_span,
            ode_solver=cfg.ode_solver,
            system=cfg.system,
            rngs=rngs,
        )

        self.phys_param_ranges = cfg.phys_param_ranges

        assert len(self.phys_param_ranges) == num_phys_params, (
            f"Number of physics parameters should be consistent. Received {len(self.phys_param_ranges)} ranges but {cfg.system} takes {num_phys_params} parameters."
        )
        self.mask = self.decode.mask

        self.alpha = cfg.alpha
        self.beta = cfg.beta
        self.gamma = cfg.gamma

    def __call__(self, x: jax.Array, key: jax.Array) -> Distribution:
        """Reconstruct the input"""
        phys_stats, aux_stats, _ = self.encode(x)
        phys_latents, aux_latents = self.sample_latents(phys_stats, aux_stats, key)
        init_y = x[:, 0].reshape(-1, 1)

        return self.decode(phys_latents, aux_latents, init_y)

    def priors(self, batch_size: int) -> tuple[list[Distribution], list[Distribution]]:
        """
        Return the prior distributions for this model

        Parameters
        ----------
        batch_size : int
            The batch_size

        Return
        ------
        phys_prior : list[Distribution]
            The priors over physics params
        aux_prior : list[Distribution]
            The priors over the auxiliary latents
        """
        phys_priors = []
        for start, end in self.phys_param_ranges:
            phys_priors.append(
                (
                    jnp.ones((batch_size, 1)) * (start + end) / 2,
                    jnp.ones((batch_size, 1)) * 2 * jnp.log(0.866 * (end - start)),
                )
            )

        aux_priors = []
        for _ in range(self.decode.num_pre_funcs + self.decode.num_post_funcs):
            aux_priors.append(
                (
                    jnp.zeros((batch_size, 1)),
                    jnp.zeros((batch_size, 1)),
                )
            )

        return phys_priors, aux_priors

    def generate_physics(self, params: list[jax.Array], init_y: jax.Array) -> jax.Array:
        """
        Using the models physics ODE/PDE and the provided parameters generate the solution to the ODE/PDE.

        Parameters
        ----------
        params : list[Array]
            The physics model parameters
        init_y : Array
            The system initial conditions

        Return
        ------
        solution : Array
            The solution at the resolution of inidividual timesteps
        """

        def ODE(_t: RealScalarLike, y: PyTree, _args: PyTree):
            return self.decode.physics_model(y, *params)

        init_cond = jnp.pad(init_y, ((0, 0), (0, self.ode_dim - self.num_grid_points)))

        y_seq = diffeqsolve(
            ODETerm(ODE),
            self.decode.solver(),
            t0=0.0,
            t1=self.integral_len,
            dt0=self.integral_step,
            y0=init_cond,
            saveat=SaveAt(ts=self.t_span),
        ).ys
        assert y_seq is not None, "Failed to solve ODE"

        trim_y_seq = y_seq[
            jnp.arange(0, self.integral_len, self.integral_level),
            :,
            : self.num_grid_points,
        ]
        perm_y = jnp.permute_dims(trim_y_seq, (1, 0, 2))

        return jnp.reshape(perm_y, (-1, self.data_dim))

    def sample_latents(
        self,
        phys_stats: list[Distribution],
        aux_stats: list[Distribution],
        key: jax.Array,
    ) -> tuple[list[jax.Array], list[jax.Array]]:
        """
        Given the statistics for the physical and auxiliary latents draw samples

        Parameters
        ----------
        phys_stats : list[Distribution]
            The distribution statistics for the physical latents
        aux_stats : list[Distribution]
            The distribution statistics for the auxiliary latents
        key : PRNG
            The random key to use

        Returns
        -------
        phys_latents : list[jax.Array]
            The sampled physical latents
        aux_latents : list[jax.Array]
            The sampled auxiliary latents
        """
        phys_latents = []
        for stat, param_range in zip(phys_stats, self.phys_param_ranges):
            key, sample_key = jax.random.split(key, 2)

            samples = stat[0] + (
                jax.random.normal(sample_key, stat[0].shape) * jnp.exp(stat[1])
            )

            # scaled_param in [param_range[0]/2, 2*param_range[1]]
            scaled_param = (
                nnx.sigmoid(samples) * ((2 * param_range[1]) - (param_range[0] / 2))
            ) + (param_range[0] / 2)

            phys_latents.append(scaled_param)

        aux_latents = []
        for stat in aux_stats:
            key, sample_key = jax.random.split(key, 2)
            aux_latents.append(
                stat[0]
                + jax.random.normal(sample_key, stat[0].shape) * jnp.exp(stat[1])
            )

        return phys_latents, aux_latents
