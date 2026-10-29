from functools import reduce
from typing import Callable, Optional

import jax
import jax.numpy as jnp
from diffrax import AbstractSolver, ODETerm, SaveAt, diffeqsolve
from diffrax._custom_types import RealScalarLike
from flax import nnx
from jaxtyping import PyTree

from models.ddim import compute_betas
from models.mlp import make_mlp
from models.modules.acts import ACTS
from models.phys_vae import gen_selections
from models.physics_models import Operator, get_physics_model, get_solver


class SinusoidalEmbedding:
    def __init__(self, dim):
        self.dim = dim

    def __call__(self, t):
        half_dim = self.dim // 2
        emb = jnp.log(10000.0) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim) * -emb)
        emb = (1000 * t[:, None]) * emb[None, :]
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=1)
        return emb


class PhysUNet2D(nnx.Module):
    """TODO: Add docstring"""

    def __init__(
        self,
        aux_funcs: list[dict],
        unmixer: dict,
        phys_funcs: list[dict],
        pre_funcs: list[dict],
        post_funcs: list[dict],
        phys_param_ranges: list[tuple[float, float]],
        t_span: jax.Array,
        t1: float,
        dt: float,
        num_grid_points: int,
        pde_dim: int,
        time_dim: int,
        physics_model: Operator,
        solver: Callable[[], AbstractSolver],
        rngs: nnx.Rngs,
        embed_dim: int = 128,
    ):
        # Constants
        num_pre_funcs = len(pre_funcs)
        num_post_funcs = len(post_funcs)
        self.pre_combs = 2**num_pre_funcs
        self.post_combs = 2**num_post_funcs
        self.pre_selections, self.post_selections = gen_selections(
            num_pre_funcs, num_post_funcs
        )

        ## PDE solver constants
        self.t_span = t_span
        self.t1 = t1
        self.dt = dt
        self.num_grid_points = num_grid_points
        self.pde_dim = pde_dim
        self.time_dim = time_dim
        self.data_dim = num_grid_points * time_dim
        # Diffusion step embedding width; decoupled from data_dim so the
        # embedding MLP and head inputs stay small (and grids like burgers
        # cannot silently explode the parameter count)
        self.embed_dim = embed_dim

        self.physics_model = physics_model
        self.solver = solver

        # Modules
        # Diffusion step embedding
        self.time_proj = SinusoidalEmbedding(dim=self.embed_dim)
        self.time_embed = nnx.Sequential(
            nnx.Linear(self.embed_dim, self.embed_dim * 4, rngs=rngs),
            ACTS["swish"],
            nnx.Linear(self.embed_dim * 4, self.embed_dim, rngs=rngs),
        )

        # Auxiliary functions
        aux_dims = []
        self.aux_funcs = nnx.List()
        for aux_cfg in aux_funcs:
            # X*T + E -> Z
            aux_func = make_mlp(
                self.data_dim + self.embed_dim,
                aux_cfg["dims"],
                aux_cfg["latent_dim"],
                aux_cfg["act"],
                rngs=rngs,
            )
            self.aux_funcs.append(aux_func)
            aux_dims.append(aux_cfg["latent_dim"])

        # Denoiser to known operator solutions
        latent_dims = reduce(lambda x, y: x + y, aux_dims)
        # Z + X*T + E -> X*T
        self.unmixer = make_mlp(
            latent_dims + self.data_dim + self.embed_dim,
            unmixer["dims"],
            self.data_dim,
            unmixer["act"],
            rngs=rngs,
        )

        # Physic parameter functions
        assert len(phys_funcs) == len(phys_param_ranges), (
            f"Number of physics model parameters must match number of parameter estimators, {len(phys_param_ranges)} != {len(phys_funcs)}"
        )
        self.phys_funcs = nnx.List()
        for phys_cfg, param_range in zip(phys_funcs, phys_param_ranges):
            min, max = param_range
            # X*T -> Z
            phys_func = nnx.Sequential(
                make_mlp(
                    self.data_dim,
                    phys_cfg["dims"],
                    phys_cfg["latent_dim"],
                    phys_cfg["act"],
                    rngs=rngs,
                ),
                nnx.sigmoid,
                # Rescale by the exact configured range; defaults bound so each
                # closure captures its own (lo, hi) rather than the loop vars
                lambda x, lo=min, hi=max: x * (hi - lo) + lo,
            )
            self.phys_funcs.append(phys_func)

        # Perturbations
        ## PDE perturbations
        self.pre_funcs = nnx.List()
        for func_cfg in pre_funcs:
            # Z + X*D + t -> dy
            func = make_mlp(
                func_cfg["latent_dim"] + self.pde_dim + 2,
                func_cfg["dims"],
                self.num_grid_points,
                func_cfg["act"],
                rngs=rngs,
            )
            self.pre_funcs.append(func)

        ## Post PDE perturbations
        self.post_funcs = nnx.List([])
        for func_cfg in post_funcs:
            # Z + X*T -> X*(T-1)
            func = make_mlp(
                func_cfg["latent_dim"] + self.data_dim,
                func_cfg["dims"],
                self.data_dim,
                func_cfg["act"],
                rngs=rngs,
            )
            self.post_funcs.append(func)

    def _embed(
        self, x: jax.Array, tau: jax.Array
    ) -> tuple[list[jax.Array], list[jax.Array], jax.Array]:
        """
        Compute physics signal, latents, and reconstruction primer all of which is the same regardless of training or inference.

        Parameters
        ----------
        x : (B, T, X, 1)
            Batched data input
        tau : (B)
            Current diffusion step

        Returns
        -------
        aux_latents : list[(B, Z)]
            List of auxiliary latents
        phys_params : list[(B, Z)]
            List of physic parameters
        phys_sig : (B, X*T)
            Estimated known physics signal
        """
        # x : (B, T, X, 1)
        h = x.reshape(-1, self.data_dim)

        t_emb = self.time_proj(tau)
        t_emb = self.time_embed(t_emb)

        # aux_latent : (B, Z)
        aux_latents = []
        for aux_func in self.aux_funcs:
            aux_latents.append(aux_func(jnp.concatenate([h, t_emb], axis=-1)))

        # full_aux : (B, Z1 + ... + Zn)
        full_aux = jnp.concatenate(aux_latents, axis=-1)

        # phys_sig : (B, X*T)
        phys_sig = self.unmixer(jnp.concatenate([h, full_aux, t_emb], axis=-1))

        # phys_latent : (B, Z)
        phys_params = []
        for phys_func in self.phys_funcs:
            phys_params.append(phys_func(phys_sig))

        return aux_latents, phys_params, phys_sig

    def _generate(
        self,
        x: jax.Array,
        aux_latents: list[jax.Array],
        phys_params: list[jax.Array],
        pre_combs: int = 1,
        post_combs: int = 1,
        pre_selections: Optional[jax.Array] = None,
        post_selections: Optional[jax.Array] = None,
    ) -> jax.Array:
        """
        TODO: write meaningful docstring

        Parameters
        ----------
        x : Array (B, T, X, 1) padded
            The input noisy sample
        aux_latents : Array (B, Z)
            The auxiliary latents
        phys_params : Array (B, Z)
            The estimated physics parameters
        pre_combs : int = 1
            The number of pre-ODE function combinations
        post_combs : int = 1
            The number of post-ODE function combinations
        pre_selections : Array (num_pre_funcs, pre_combs)
            A binary array of which combinations to apply a function.
        post_selections : Array (num_post_funcs, pre_combs * post_combs)
            A binary array of which combinations to apply a function.

        Returns
        -------
        phys_x_hat : Array (m, B, X*T)
            The estimated clean sample incorporating physics knowledge
        """

        if pre_selections is None or post_selections is None:
            assert pre_combs == 1 and post_combs == 1, (
                f"If selections are not specified combinations must be 1, {pre_combs} != 1 or {post_combs} != 1."
            )

            pre_selections = jnp.ones((len(self.pre_funcs), 1))
            post_selections = jnp.ones((len(self.post_funcs), 1))
        else:
            assert (
                pre_selections.shape == (len(self.pre_funcs), pre_combs)
                or len(self.pre_funcs) == 0
            ), (
                f"Shape of pre_selections must match number of pre-ODE functions and pre_combinations, {pre_selections.shape} != {(len(self.pre_funcs), pre_combs)} and {len(self.pre_funcs)} != 0."
            )
            assert (
                post_selections.shape == (len(self.post_funcs), pre_combs * post_combs)
                or len(self.post_funcs) == 0
            ), (
                f"Shape of post_selections must match number of post-ODE functions and post_combinations, {post_selections.shape} != {(len(self.post_funcs), pre_combs * post_combs)} and {len(self.post_funcs)} != 0."
            )

        def PDE(t: RealScalarLike, y: PyTree, args: PyTree):
            """Internal PDE to be solved during the generation process."""
            # y : (B*n, X*D)
            assert len(y.shape) == 2, f"expected (B, X*D) but received {y.shape}"
            assert y.shape[1] == self.pde_dim, (
                f"Expected (B, {self.pde_dim}) but received {y.shape}"
            )

            ys = y.reshape(pre_combs, -1, self.pde_dim)

            # y_dot_phys : (n, B, X*D)
            ys_dot_phys = jax.vmap(
                self.physics_model, in_axes=(None, 0, None), out_axes=0
            )(t, ys, args)

            # ys_dot_aux : (n, B, X*D)
            ys_dot_aux = jnp.zeros_like(ys_dot_phys)
            for z_aux, pre_func, selections in zip(
                aux_latents, self.pre_funcs, pre_selections
            ):
                # y : (B, D), apply : (1)
                def apply_func(y, apply):
                    t_vec = jnp.ones_like(y)[:, 0, None] * t
                    return apply * jnp.pad(
                        pre_func(
                            jnp.concatenate(
                                [z_aux, y, jnp.sin(t_vec), jnp.cos(t_vec)], axis=-1
                            )
                        ),
                        ((0, 0), (self.pde_dim - self.num_grid_points, 0)),
                    )

                ys_dot_aux += jax.vmap(apply_func, in_axes=(0, 0), out_axes=0)(
                    ys, selections
                )

            # ys_dot: (B*n, X*D)
            return jnp.reshape(ys_dot_phys + ys_dot_aux, (-1, self.pde_dim))

        # init_cond : ^(B*n, X*D)
        init_cond = jnp.tile(
            jnp.pad(
                x[:, 0].reshape(-1, self.num_grid_points),
                ((0, 0), (0, self.pde_dim - self.num_grid_points)),
            ),
            reps=(pre_combs, 1),
        )

        # y_seq : ^(T, B*n, X*D)
        y_seq = diffeqsolve(
            ODETerm(PDE),
            self.solver(),
            t0=0.0,
            t1=self.t1,
            dt0=self.dt,
            y0=init_cond,
            args=phys_params,
            saveat=SaveAt(ts=self.t_span),
        ).ys
        assert y_seq is not None, "Failed to solve PDE"

        # ^(n, B, X*T)
        phys_x_hat = jnp.reshape(
            # ^(T, B*n, X*D) => ^(B*n, T, X)
            jnp.permute_dims(y_seq[:, :, : self.num_grid_points], (1, 0, 2)),
            (pre_combs, -1, self.data_dim),
        )

        # phys_x_hat : (m, B, X*T)
        phys_x_hat = jnp.tile(phys_x_hat, [post_combs, 1, 1])

        for z_aux, post_func, selections in zip(
            aux_latents[
                len(self.pre_funcs) : len(self.pre_funcs) + len(self.post_funcs)
            ],
            self.post_funcs,
            post_selections,
        ):
            # z_aux : (B, Z), phys : (B, X*T) -> (B, X*T)
            def apply_func(phys, apply):
                return (1 - apply) * phys + apply * post_func(
                    jnp.concatenate([z_aux, phys], axis=-1)
                )

            # phys_x_hat : (m, B, X*T)
            phys_x_hat = jax.vmap(apply_func, in_axes=(0, 0), out_axes=0)(
                phys_x_hat, selections
            )

        return phys_x_hat

    def forward(self, x: jax.Array, tau: jax.Array) -> jax.Array:
        """
        TODO: Add docstring

        Parameters
        ----------
        x : (B, T, X, 1)
            Batched noised data
        tau : (B)
            Current diffusion step

        Returns
        -------
        x_hat : Array (B, T, X, 1)
            Estimation of the clean x
        """
        aux_latents, phys_latents, _ = self._embed(x, tau)

        # x_hat : (1, B, X*T)
        x_hat = self._generate(x, aux_latents, phys_latents)

        assert len(x_hat.shape) == 3, (
            "Internal issue with generation should return combination solution."
        )
        assert x_hat.shape[0] == 1, (
            "Internal issue with generation should return single combination."
        )

        return jnp.reshape(x_hat[0], x.shape)

    def full_forward(
        self, x: jax.Array, tau: jax.Array
    ) -> tuple[jax.Array, jax.Array, list[jax.Array]]:
        """
        TODO: Add docstring

        Parameters
        ----------
        x : Array (B, T, X, 1) padded
            Batched data input
        tau : Array (B)
            Current diffusion step

        Returns
        -------
        x_hat : Array (m, B, T, X, 1)
            Estimation of clean signal using combinations of auxiliary functions
        phys_sig : Array (B, T, X, 1)
            Estimation of the known physics signal
        phys_params : list[Array (B, Z)]
            Estimation of physics parameters
        """
        aux_latents, phys_params, phys_sig = self._embed(x, tau)

        # x_hat : (m, B, X*T)
        x_hat = self._generate(
            x,
            aux_latents,
            phys_params,
            self.pre_combs,
            self.post_combs,
            self.pre_selections,
            self.post_selections,
        )

        assert len(x_hat.shape) == 3, (
            "Internal issue with generation should return combination solution."
        )
        assert x_hat.shape[0] == self.pre_combs * self.post_combs, (
            "Internal issue with generation should return all combinations."
        )

        # x_hat: (m, B, T, X, 1), phys_sig : (B, T, X, 1), phys_params : (B, Z)
        return (
            jnp.reshape(x_hat, (self.pre_combs * self.post_combs,) + x.shape),
            jnp.reshape(phys_sig, x.shape),
            phys_params,
        )


class PAID(nnx.Module):
    """
    Physics aligned interpretable diffusion (PAID) model.
    """

    def __init__(
        self,
        pde: dict,
        aux_funcs: list[dict],
        unmixer: dict,
        phys_funcs: list[dict],
        pre_funcs: list[dict],
        post_funcs: list[dict],
        num_timesteps: int,
        beta_schedule: str,
        beta_start: float,
        beta_end: float,
        disc_reg: float,
        phys_reg: float,
        aug_reg: float,
        dataset_config: dict,
        embed_dim: int = 128,
        condition: int = 5,
        jump: int = 10,
        pad: list[tuple] = [(0, 0), (0, 0)],
        *,
        rngs: nnx.Rngs,
    ) -> None:
        system = dataset_config["system"]
        length = dataset_config["length"]
        time = dataset_config["time"]
        x_grid = dataset_config["x_grid"]
        t_grid = dataset_config["t_grid"]
        # Diffusion constants
        self.num_timesteps = num_timesteps
        self.betas = compute_betas(
            beta_schedule, beta_start, beta_end, self.num_timesteps
        )

        # Training constants
        self.disc_reg = disc_reg
        self.phys_reg = phys_reg
        self.aug_reg = aug_reg

        # Physics constants + functions
        # The PDE system is defined by dataset_config: the data is generated
        # on an even x_grid x t_grid discretisation of a length x time domain,
        # so the frame step is derived here rather than configured separately.
        self.integral_level = pde["integral_level"]
        self.time_dim = t_grid

        frame_dt = time / float(t_grid - 1)
        self.dt = frame_dt / float(self.integral_level)
        self.t1 = (self.time_dim - 1) * self.dt
        self.t_span = jnp.linspace(0.0, self.t1, self.time_dim)

        self.num_grid_points = x_grid
        ds = 0.0
        if self.num_grid_points > 1:
            ds = length / (self.num_grid_points - 1)

        physics_model, _, num_derivatives = get_physics_model(system)
        self.pde_dim = self.num_grid_points * num_derivatives
        self.physics_model = physics_model(jnp.array(ds), "periodic")
        self.solver = get_solver(pde["solver"])

        assert self.physics_model is not None
        assert self.solver is not None

        # Denoiser
        self.denoiser = PhysUNet2D(
            aux_funcs,
            unmixer,
            phys_funcs,
            pre_funcs,
            post_funcs,
            pde["phys_param_ranges"],
            self.t_span,
            self.t1,
            self.dt,
            self.num_grid_points,
            self.pde_dim,
            self.time_dim,
            self.physics_model,
            self.solver,
            embed_dim=embed_dim,
            rngs=rngs,
        )

        # Additional constants
        self.condition = condition
        self.jump = jump
        self.phys_param_ranges = pde["phys_param_ranges"]
        self.pad = pad

    def __call__(self, x: jax.Array, key: jax.Array) -> tuple[jax.Array, jax.Array]:
        """
        Sample the conditioned DDIM trajectory and estimate the physics
        coefficient, mirroring the PhysVAE ``__call__(x, key)`` protocol.

        The denoising diffusion sampling runs to completion over
        ``ceil(num_timesteps / jump)`` DDIM steps: the denoiser predicts the
        clean sample (mean estimation), the noise is implied by ``x_t`` and
        ``x_0``, and the conditioned frames are carried clean at every step.
        The coefficient is estimated from the final step of the trajectory:
        the generated sample is noised back a single diffusion step and the
        embedder's physics parameters are read off it (the unmixer is never
        trained on the clean section, so the estimate comes from the
        trajectory's own sample at ``tau = 1``).

        Parameters
        ----------
        x : Array (B, T, X) or (B, T, X, 1)
            Batched data; the first ``condition`` frames are the condition
        key : PRNG
            Random key for the initial noise and the coefficient re-noising

        Returns
        -------
        x_0 : Array
            The generated sample, channel axis dropped if the input was 3-D
        coeff : Array (B, 1)
            The estimated physics coefficient
        """
        expanded = x.ndim < 4
        if expanded:
            x = jnp.expand_dims(x, -1)

        alphas = jnp.cumprod(1.0 - self.betas)

        key, noise_key = jax.random.split(key)
        # The conditioned frames are never noise; the rest starts pure noise
        x_t = jnp.concatenate(
            [
                x[:, : self.condition],
                jax.random.normal(noise_key, x.shape)[:, self.condition :],
            ],
            axis=1,
        )

        t_last = self.num_timesteps - 1
        # Ceiling division so the final step lands exactly on tau = 0
        num_steps = (self.num_timesteps + self.jump - 1) // self.jump

        def ddim_step(i: jax.Array, x_t: jax.Array) -> jax.Array:
            t = t_last - i * self.jump
            t_next = jnp.maximum(t - self.jump, 0)

            tau = jnp.full((x.shape[0],), t, dtype="int32")
            x_hat, _, _ = self.denoiser.full_forward(x_t, tau)
            x_0 = x_hat[-1]

            alpha_t = alphas[t]
            alpha_next = alphas[t_next]

            # DDIM with the noise implied by x_t and x_0 (deterministic)
            eps = (x_t - x_0 * jnp.sqrt(alpha_t)) / jnp.sqrt(1.0 - alpha_t)
            x_next = x_0 * jnp.sqrt(alpha_next) + eps * jnp.sqrt(1.0 - alpha_next)
            x_next = jax.lax.cond(
                t_next == 0,
                lambda: x_0,
                lambda: x_next,
            )

            # Conditioned frames stay clean throughout
            return jnp.concatenate(
                [x_t[:, : self.condition], x_next[:, self.condition :]], axis=1
            )

        x_0 = jax.lax.fori_loop(0, num_steps, ddim_step, x_t)

        # Coefficient estimate: noise the sample back one step and read the
        # embedder's physics parameters at tau = 1
        key, noise_key = jax.random.split(key)
        alpha_1 = alphas[1]
        x_1 = x_0 * jnp.sqrt(alpha_1) + jax.random.normal(
            noise_key, x_0.shape
        ) * jnp.sqrt(1.0 - alpha_1)
        tau = jnp.ones((x.shape[0],), dtype="int32")
        _, phys_params, _ = self.denoiser._embed(x_1, tau)
        coeff = phys_params[0]

        if expanded:
            x_0 = jnp.squeeze(x_0, axis=-1)

        return x_0, coeff

    def generate_physics(self, params: list[jax.Array], init_y: jax.Array) -> jax.Array:
        """
        TODO: generate docstring

        Parameters
        ----------
        params : list[Array (B, Z)]
            List of physics parameters
        init_y : Array (B, X)
            Initial condition

        Returns
        -------
        gen_phys : Array (B, X*T)
            Physics generated by the provided parameters
        """
        # init_cond : ^(B*n, X*D)
        init_cond = jnp.pad(
            init_y,
            ((0, 0), (0, self.pde_dim - self.num_grid_points)),
        )

        # y_seq : ^(T, B, X*D)
        y_seq = diffeqsolve(
            ODETerm(self.physics_model),
            self.solver(),
            t0=0.0,
            t1=self.t1,
            dt0=self.dt,
            y0=init_cond,
            args=params,
            saveat=SaveAt(ts=self.t_span),
        ).ys
        assert y_seq is not None, "Failed to solve PDE"

        # ^(B, X*T)
        gen_phys = jnp.reshape(
            # ^(T, B, X*D) => ^(B, X*T)
            jnp.permute_dims(y_seq[:, :, : self.num_grid_points], (1, 0, 2)),
            (-1, self.time_dim * self.num_grid_points),
        )

        return gen_phys
