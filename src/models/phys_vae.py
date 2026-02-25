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

    Parameters
    ----------
    y : (2,)
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


PHYS_MODELS = {"pendulum": pendulum_model}

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

    def __init__(self, cfg: PhysEncoderConfig, time_dim: int, rngs: nnx.Rngs) -> None:
        self.time_dim = time_dim

        # Setup auxiliary functions
        self.aux_latent_dims = []
        self.aux_funcs = nnx.List()
        for aux_cfg in cfg.aux_funcs:
            self.aux_latent_dims.append(aux_cfg.latent_dim)

            # Assumes all models are MLPs
            aux_func = make_mlp(
                time_dim, aux_cfg.dims, 2 * aux_cfg.latent_dim, aux_cfg.act, rngs=rngs
            )

            self.aux_funcs.append(aux_func)

        latent_dims = reduce(lambda x, y: x + y, self.aux_latent_dims)

        self.unmixer = make_mlp(
            latent_dims + time_dim,
            cfg.unmixer.dims,
            cfg.unmixer.latent_dim,
            cfg.unmixer.act,
            rngs=rngs,
        )
        self.residual = cfg.unmixer.residual

        self.phys_funcs = nnx.List()
        for phys_cfg in cfg.phys_funcs:
            # Assumes all models are MLPs
            phys_func = make_mlp(
                time_dim,
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
    time_dim : int
        The time dimension, equivalently the number of timesteps
    t1 : int
        The final timestep of the simulation
    dt : float
        The length of the timestep
    integral_level : int
        The solver substeps
    t_span : Array (T)
        The steps to calculate the ODE on
    ode_solver : str
        The ODE solver name
    system : str
        The underlying physical system
    rngs : nnx.Rngs
        The model RNGs
    """

    def __init__(
        self,
        cfg: PhysDecoderConfig,
        time_dim: int,
        t1: int,
        dt: float,
        integral_level: int,
        t_span: jax.Array,
        ode_solver: str,
        system: str,
        rngs: nnx.Rngs,
    ) -> None:
        self.time_dim = time_dim
        self.t1 = t1
        self.dt = dt
        self.integral_level = integral_level
        self.t_span = t_span

        self.x_log_sigma = cfg.x_log_sigma

        self.pre_funcs = nnx.List()
        self.post_funcs = nnx.List()
        for func_cfg in cfg.pre_funcs:
            # Assume all models are MLPs
            # TODO: what if y is not dim(2)
            # z + y (2) + t -> dy
            func = make_mlp(
                func_cfg.latent_dim + 2 + 1, func_cfg.dims, 1, func_cfg.act, rngs=rngs
            )
            self.pre_funcs.append(func)
        self.pre_combs = 2 ** len(self.pre_funcs)
        self.pre_selections = gen_selections(len(self.pre_funcs))

        phy_latent_dim = 1
        for func_cfg in cfg.post_funcs:
            func = make_mlp(
                func_cfg.latent_dim + phy_latent_dim,
                func_cfg.dims,
                self.time_dim,
                func_cfg.act,
                rngs=rngs,
            )
            self.post_funcs.append(func)
            phy_latent_dim = self.time_dim
        self.post_combs = 2 ** len(self.post_funcs)
        self.post_selections = gen_selections(len(self.post_funcs))

        self.physics_model = PHYS_MODELS.get(system, None)
        if self.physics_model is None:
            raise NotImplementedError(f"Physics model {system} is not implemented.")

        self.solver = SOLVERS.get(ode_solver, None)
        if self.solver is None:
            raise NotImplementedError(f"ODE solver {system} is not implemented.")

        self.mask = gen_mask(self.pre_combs + self.post_combs)

    def __call__(
        self,
        phys_params: list[jax.Array],
        latents: list[jax.Array],
        init_y: jax.Array,
        train=False,
    ) -> Distribution:
        """
        Given physical and auxiliary latents decode the full series based on the initial condition.

        Parameters
        ----------
        phys_params : list[jax.Array] [(B, Z)]
            The physical latents
        latents : list[jax.Array] [(B, Z)]
            The auxiliary latents
        init_y : Array (B, 1)
            The initial condition
        train : bool = False
            Whether the model is training or evaluating. Training returns all combinations of auxiliary functions which is unecessary for inference

        Returns
        -------
        series : Distribution (m, B, T) or (B, T)
            The statistics for the series with the mean and log variance stacked in the first axis
        """
        batch_size = phys_params[0].shape[0]
        assert all([param.shape[0] == batch_size for param in phys_params]), (
            "All parameters should be the same batch size"
        )

        def ODE(t: RealScalarLike, y: PyTree, _args: PyTree):
            """Internal ODE to be solved during the generation process."""
            # y : (B*n, 2)
            assert len(y.shape) == 2, f"expected (B*n, 2) but received {y.shape}"
            assert y.shape[1] == 2, f"expected (B*n, 2) but received {y.shape}"
            if train:
                # ys : (n, B, 2)
                ys = jnp.reshape(y, (self.pre_combs, -1, 2))
            else:
                # ys : (1, B, 2)
                ys = y[:, [0, 1]].reshape(1, -1, 2)

            # ys_dot_phys : (n, B, 2)
            ys_dot_phys = jax.vmap(self.physics_model, in_axes=(0, None), out_axes=0)(
                ys, *phys_params
            )

            if train:
                # ys_dot_aux : (n, B, 2)
                ys_dot_aux = jnp.zeros((self.pre_combs, batch_size, 1))
            else:
                # ys_dot_aux : (1, B, 2)
                ys_dot_aux = jnp.zeros((1, batch_size, 1))

            # selections : (n, 1), z_aux : (B, l_dim)
            for z_aux, pre_func, selections in zip(
                latents[: len(self.pre_funcs)], self.pre_funcs, self.pre_selections
            ):
                # y : (B, 2), apply : (1,)
                def apply_func(y, apply):
                    return apply * jnp.concatenate(
                        [
                            jnp.zeros((batch_size, 1)),
                            pre_func(
                                jnp.concatenate(
                                    [z_aux, y, jnp.ones_like(y)[:, 0, None] * t], axis=1
                                )
                            ),
                        ],
                        axis=1,
                    )

                # ys_dot_aux : (n, B, 2)
                ys_dot_aux += jax.vmap(apply_func, in_axes=(0, 0), out_axes=0)(
                    ys, selections
                )

            # ys_dot : (B*n, 2)
            return jnp.reshape(ys_dot_phys + ys_dot_aux, (-1, 2))

        if train:
            init_cond = jnp.tile(
                jnp.concatenate([init_y, jnp.zeros_like(init_y)], axis=1),
                reps=(self.pre_combs, 1),
            )
        else:
            init_cond = jnp.concatenate([init_y, jnp.zeros_like(init_y)], axis=1)

        y_seq = diffeqsolve(
            ODETerm(ODE),
            self.solver(),
            t0=0.0,
            t1=self.t1,
            dt0=self.dt,
            y0=init_cond,
            saveat=SaveAt(ts=self.t_span),
        ).ys
        # y_seq : (S, B*n, 2)
        assert y_seq is not None, "Failed to solve ODE"

        # trim_y_seq : (T, B*n, 2)
        trim_y_seq = y_seq[jnp.arange(0, self.t1, self.integral_level), :, :]

        # x_means : (m, B, T)
        if train:
            # trim_ys : (T, n, B, 2)
            trim_ys = jnp.reshape(trim_y_seq, (self.time_dim, self.pre_combs, -1, 2))
            # x_mean : (n, B, T)
            x_mean = jnp.permute_dims(trim_ys[:, :, :, 0], (1, 2, 0))
            x_means = jnp.tile(x_mean, reps=(self.post_combs, 1, 1))
        else:
            # x_mean : (B, T)
            x_mean = trim_y_seq[:, :, 0].T
            x_means = jnp.reshape(x_mean, (1, -1, self.time_dim))

        for z_aux, post_func, selections in zip(
            latents[len(self.pre_funcs) : len(self.post_funcs)],
            self.post_funcs,
            self.post_selections,
        ):
            # x_mean : (B, T), apply : (1,)
            def apply_func(x_mean, apply):
                return (1 - apply) * x_mean + apply * jnp.concatenate(
                    [
                        jnp.zeros((batch_size, 1)),
                        post_func(jnp.concatenate([z_aux, x_mean])),
                    ]
                )

            # x_means : (m, B, T)
            x_means = jax.vmap(apply_func, in_axes=(0, 0), out_axes=0)(
                x_means, selections
            )

        if not train:
            assert x_means.shape[0] == 1
            # x_means : (B, T)
            x_means = jnp.reshape(x_means, (-1, self.time_dim))

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
        self.physics_model = PHYS_MODELS[cfg.system]

        assert len(range(0, self.integral_len, self.integral_level)) == cfg.time_dim, (
            f"Time dimension and integral steps not compatible {len(range(0, self.integral_len, self.integral_level))} != {cfg.time_dim}"
        )

        self.encode = PhysEncoder(cfg=cfg.encoder, time_dim=cfg.time_dim, rngs=rngs)
        self.decode = PhysDecoder(
            cfg=cfg.decoder,
            time_dim=cfg.time_dim,
            t1=self.integral_len,
            dt=self.integral_step,
            integral_level=cfg.integral_level,
            t_span=self.t_span,
            ode_solver=cfg.ode_solver,
            system=cfg.system,
            rngs=rngs,
        )

        self._phys_stats = cfg.phys_priors
        self._aux_stats = cfg.aux_priors

        self.mask = self.decode.mask

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
        for stat in self._phys_stats:
            phys_priors.append(
                (
                    jnp.ones((batch_size, 1)) * stat[0],
                    jnp.ones((batch_size, 1)) * stat[1],
                )
            )

        aux_priors = []
        for stat in self._aux_stats:
            aux_priors.append(
                (
                    jnp.ones((batch_size, 1)) * stat[0],
                    jnp.ones((batch_size, 1)) * stat[1],
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
            return self.physics_model(y, *params)

        init_cond = jnp.concatenate([init_y, jnp.zeros_like(init_y)], axis=1)

        y_seq = diffeqsolve(
            ODETerm(ODE),
            Euler(),
            t0=0.0,
            t1=self.integral_len,
            dt0=self.integral_step,
            y0=init_cond,
            saveat=SaveAt(ts=self.t_span),
        ).ys
        assert y_seq is not None, "Failed to solve ODE"

        return y_seq[jnp.arange(0, self.integral_len, self.integral_level), :, 0].T

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
        for stat in phys_stats:
            key, sample_key = jax.random.split(key, 2)
            phys_latents.append(
                stat[0]
                + jax.random.normal(sample_key, stat[0].shape) * jnp.exp(stat[1])
            )

        aux_latents = []
        for stat in aux_stats:
            key, sample_key = jax.random.split(key, 2)
            aux_latents.append(
                stat[0]
                + jax.random.normal(sample_key, stat[0].shape) * jnp.exp(stat[1])
            )

        return phys_latents, aux_latents
