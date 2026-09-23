from functools import reduce
from typing import Callable, Optional

import jax
import jax.numpy as jnp
from diffrax import AbstractSolver, ODETerm, SaveAt, diffeqsolve
from diffrax._custom_types import RealScalarLike
from flax import nnx
from jaxtyping import PyTree

from diffuse.neural_network.blocks import ResnetBlock
from diffuse.neural_network.nn import CondUNet
from models.ddim import compute_betas
from models.mlp import make_mlp
from models.modules.acts import ACTS
from models.phys_vae import gen_selections
from models.physics_models import Operator, get_physics_model, get_solver
from models.utils import remove_padding


class PhysUNet2D(nnx.Module):
    """TODO: Add docstring"""

    def __init__(
        self,
        unet_config: dict,
        pde_config: dict,
        aux_funcs: list[dict],
        unmixer: dict,
        phys_funcs: list[dict],
        pre_funcs: list[dict],
        post_funcs: list[dict],
        integral_level: int,
        t_span: jax.Array,
        t1: float,
        dt: float,
        num_grid_points: int,
        pde_dim: int,
        time_dim: int,
        physics_model: Operator,
        solver: Callable[[], AbstractSolver],
        pad: list[tuple[int, int]],
        rngs: nnx.Rngs,
    ):
        # Constants
        self.pad = pad

        num_pre_funcs = len(pre_funcs)
        num_post_funcs = len(post_funcs)
        self.pre_combs = 2**num_pre_funcs
        self.post_combs = 2**num_post_funcs
        self.pre_selections, self.post_selections = gen_selections(
            num_pre_funcs, num_post_funcs
        )

        ## PDE solver constants
        self.integral_level = integral_level
        self.t_span = t_span
        self.t1 = t1
        self.dt = dt
        self.num_grid_points = num_grid_points
        self.pde_dim = pde_dim
        self.time_dim = time_dim

        self.dimension = 2 if self.num_grid_points > 1 else 1

        self.physics_model = physics_model
        self.solver = solver

        # Modules
        # temporal Unet
        unet_config["activation"] = ACTS[
            unet_config["activation"]
        ]  # Convert to function
        self.unet = CondUNet(**unet_config, dimension=self.dimension, rngs=rngs)
        unet_latent_features = unet_config["ch_mult"][-1] * unet_config["ch"]

        # Auxiliary functions
        aux_dims = []
        halving = 2 ** (len(unet_config["ch_mult"]) - 1)

        if self.dimension == 2:
            flat_dim_base = ((self.num_grid_points + sum(self.pad[1])) // halving) * (
                (self.time_dim + sum(self.pad[0])) // halving
            )
        elif self.dimension == 1:
            flat_dim_base = (self.time_dim + sum(self.pad[0])) // halving
        else:
            raise NotImplementedError(
                f"Dimension of {self.dimension} is not supported."
            )

        self.aux_funcs = nnx.List()
        for aux_cfg in aux_funcs:
            flat_dim = flat_dim_base * aux_cfg["features"]
            aux_func = nnx.Sequential(
                ResnetBlock(
                    in_channels=unet_latent_features,
                    out_channels=aux_cfg["features"],
                    activation=ACTS[aux_cfg["act"]],
                    embedding_dim=aux_cfg.get("time_dim"),
                    dropout=aux_cfg["dropout"],
                    dropout_rate=aux_cfg["dropout_rate"],
                    dimension=self.dimension,
                    rngs=rngs,
                ),
                ACTS[aux_cfg["act"]],
                lambda x: x.reshape(x.shape[0], -1),
                nnx.Linear(flat_dim, aux_cfg["latent_dim"], rngs=rngs),
            )
            self.aux_funcs.append(aux_func)
            aux_dims.append(aux_cfg["latent_dim"])

        # Known physics signal estimators
        latent_dims = reduce(lambda x, y: x + y, aux_dims)
        self.unmixer_conv = nnx.Conv(
            in_features=latent_dims + 1,
            out_features=unmixer["dims"][0],
            kernel_size=(3,) * self.dimension,
            strides=(1,) * self.dimension,
            padding=(1,) * self.dimension,
            rngs=rngs,
        )
        self.unmixer = nnx.List([])
        for in_features, out_features in zip(unmixer["dims"][:-1], unmixer["dims"][1:]):
            self.unmixer.append(
                ResnetBlock(
                    in_channels=in_features,
                    out_channels=out_features,
                    activation=ACTS[unmixer["act"]],
                    embedding_dim=unmixer.get("time_dim"),
                    dropout=unmixer["dropout"],
                    dropout_rate=unmixer["dropout_rate"],
                    dimension=self.dimension,
                    rngs=rngs,
                )
            )
            in_features = out_features
        self.unmixer_out = nnx.Conv(
            in_features=unmixer["dims"][-1],
            out_features=1,
            kernel_size=(3,) * self.dimension,
            strides=(1,) * self.dimension,
            padding=(1,) * self.dimension,
            rngs=rngs,
        )

        # Physic parameter functions
        assert len(phys_funcs) == len(pde_config["phys_param_ranges"]), (
            f"Number of physics model parameters must match number of parameter estimators, {len(pde_config['phys_param_ranges'])} != {len(phys_funcs)}"
        )
        self.phys_funcs = nnx.List()
        for phys_cfg, param_range in zip(phys_funcs, pde_config["phys_param_ranges"]):
            flat_dim = flat_dim_base * phys_cfg["features"]
            phys_func = nnx.Sequential(
                ResnetBlock(
                    in_channels=unet_latent_features,
                    out_channels=phys_cfg["features"],
                    activation=ACTS[phys_cfg["act"]],
                    embedding_dim=phys_cfg.get("time_dim"),
                    dropout=phys_cfg["dropout"],
                    dropout_rate=phys_cfg["dropout_rate"],
                    dimension=self.dimension,
                    rngs=rngs,
                ),
                ACTS[phys_cfg["act"]],
                lambda x: x.reshape(x.shape[0], -1),
                nnx.Linear(flat_dim, phys_cfg["latent_dim"], rngs=rngs),
                nnx.sigmoid,
                lambda x: x
                * ((2 * param_range[1]) - (param_range[0] / 2) + (param_range[0] / 2)),
            )
            self.phys_funcs.append(phys_func)

        # Perturbations
        ## PDE perturbations
        self.pre_funcs = nnx.List()
        for func_cfg in pre_funcs:
            # Z + X*D + t -> dy
            func = make_mlp(
                func_cfg["latent_dim"] + self.pde_dim + 1,
                func_cfg["dims"],
                self.num_grid_points,
                func_cfg["act"],
                rngs=rngs,
            )
            self.pre_funcs.append(func)

        ## Post PDE perturbations
        self.post_funcs = nnx.List()
        for func_cfg in post_funcs:
            func = nnx.Sequential(
                nnx.Conv(
                    in_features=2 + func_cfg["latent_dim"],
                    out_features=func_cfg["features"],
                    kernel_size=(3,) * self.dimension,
                    strides=(1,) * self.dimension,
                    padding=(1,) * self.dimension,
                    rngs=rngs,
                ),
                nnx.GroupNorm(
                    num_features=func_cfg["features"],
                    num_groups=8,
                    rngs=rngs,
                ),
                nnx.Conv(
                    in_features=func_cfg["features"],
                    out_features=1,
                    kernel_size=(3,) * self.dimension,
                    strides=(1,) * self.dimension,
                    padding=(1,) * self.dimension,
                    rngs=rngs,
                ),
            )
            self.post_funcs.append(func)

    def _embed(
        self, x: jax.Array, tau: jax.Array
    ) -> tuple[jax.Array, list[jax.Array], list[jax.Array], jax.Array]:
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
        h : (B, T, X, 1)
            UNet augmented input
        aux_latents : list[(B, Z)]
            List of auxiliary latents
        phys_params : list[(B, Z)]
            List of physic parameters
        phys_sig : (B, T, X, 1)
            Estimated known physics signal
        """
        # x : (B, T, X, 1)
        h = x

        t_emb = self.unet.time_proj(tau)
        t_emb = self.unet.time_embedding(t_emb)

        hs = []
        for block in self.unet.blocks_down:
            h = block(h, t_emb)
            hs.append(h)

        h = self.unet.block_mid(h, t_emb)

        # aux_latent : (B, Z)
        aux_latents = []
        for aux_func in self.aux_funcs:
            aux_latents.append(aux_func(h))

        # full_aux : (B, Z1 + ... + Zn)
        full_aux = jnp.concatenate(aux_latents, axis=-1)

        # phys_sig : (B, T, X, 1)
        phys_sig = self.unmixer_conv(
            jnp.concatenate(
                [
                    x,
                    jnp.tile(
                        jnp.expand_dims(
                            full_aux, axis=list(range(1, self.dimension + 1))
                        ),
                        [1, *x.shape[1 : self.dimension + 1], 1],
                    ),
                ],
                axis=-1,
            )
        )
        for layer in self.unmixer:
            phys_sig = layer(phys_sig, t_emb)

        phys_sig = self.unmixer_out(phys_sig)
        phys_h = phys_sig

        for block in self.unet.blocks_down:
            phys_h = block(phys_h, t_emb)

        phys_h = self.unet.block_mid(phys_h, t_emb)

        # phys_latent : (B, Z)
        phys_params = []
        for phys_func in self.phys_funcs:
            phys_params.append(phys_func(phys_h))

        for block in self.unet.blocks_up:
            h = jnp.concatenate([h, hs.pop()], axis=-1)
            h = block(h, t_emb)

        # h : (B, T, X, 1)
        h = self.unet.out(h)

        return h, aux_latents, phys_params, phys_sig

    def _generate(
        self,
        x: jax.Array,
        x_hat: jax.Array,
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
        x : Array (B, W, H, C) padded
            The input noisy sample
        x_hat : Array (B, W, H, C) padded
            Clean estimate without physics information
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
        phys_x_hat : Array (m, B, W, H, C)
            The estimated clean sample incorporating physics knowledge
        """

        if pre_selections is None or post_selections is None:
            assert pre_combs == 1 and post_combs == 1, (
                f"If selections are not specified combinations must be 1, {pre_combs} != 1 or {post_combs} != 1."
            )

            pre_selections = jnp.ones((len(self.pre_funcs), 1))
            post_selections = jnp.ones((len(self.pre_funcs), 1))
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
                f"Shape of post_selections must match number of post-ODE functions and post_combinations, {post_selections.shape} != {(len(self.post_funcs), pre_combs * post_combs)} and {len(self.pre_funcs)} != 0."
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
                    return apply * jnp.pad(
                        pre_func(
                            jnp.concatenate(
                                [z_aux, y, jnp.ones_like(y)[:, 0, None] * t], axis=-1
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
                remove_padding(x, self.pad)[:, 0].reshape(-1, self.num_grid_points),
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

        # ^(n, B, T, X)
        phys_x_hat = jnp.reshape(
            # ^(T, B*n, X*D) => ^(B*n, T, X)
            jnp.permute_dims(y_seq[:, :, : self.num_grid_points], (1, 0, 2)),
            (pre_combs, -1, self.time_dim, self.num_grid_points),
        )
        if self.dimension == 2:
            # phys_x_hat : ^(n, B, T, X, 1)
            phys_x_hat = jnp.expand_dims(phys_x_hat, axis=-1)

        # phys_x_hat : (m, B, T, X, 1)
        phys_x_hat = jnp.pad(
            jnp.tile(phys_x_hat, [post_combs, *((1,) * (self.dimension + 2))]),
            [(0, 0), (0, 0), *self.pad, (0, 0)],
        )

        assert x_hat.shape == phys_x_hat.shape[1:], (
            f"Both clean estimates should be the same shape, {x_hat.shape} != {phys_x_hat.shape[1:]}."
        )

        for z_aux, post_func, selections in zip(
            aux_latents[len(self.pre_funcs) : len(self.post_funcs)],
            self.post_funcs,
            post_selections,
        ):
            # z_expanded : (B, T, X, Z)
            z_expanded = jnp.tile(
                jnp.expand_dims(z_aux, axis=list(range(1, self.dimension + 1))),
                [1, *x.shape[1 : self.dimension + 1], 1],
            )

            def apply_func(phys, apply):
                return (1 - apply) * phys + apply * post_func(
                    jnp.concatenate([x_hat, phys, z_expanded], axis=-1)
                )

            # phys_x_hat : (m, B, T, X, 1)
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
        x_hat : Array (B, W, H, C) padded
            Estimation of the clean x
        """
        if self.dimension == 2:
            assert x.shape[1:] == (
                self.time_dim + sum(self.pad[0]),
                self.num_grid_points + sum(self.pad[1]),
                1,
            ), (
                f"Unexpected shape for input, {x.shape} != ({x.shape[0]}, {self.time_dim + sum(self.pad[0])}, {self.num_grid_points + sum(self.pad[1])}, 1)"
            )
        elif self.dimension == 1:
            assert x.shape[1:] == (
                self.time_dim + sum(self.pad[0]),
                1,
            ), (
                f"Unexpected shape for input, {x.shape} != ({x.shape[0]}, {self.time_dim + sum(self.pad[0])}, 1)"
            )

        x_hat, aux_latents, phys_latents, _ = self._embed(x, tau)

        # x_hat : (1, B, T, X, 1)
        x_hat = self._generate(x, x_hat, aux_latents, phys_latents)

        assert len(x_hat.shape) == 5, (
            "Internal issue with generation should return combination solution."
        )
        assert x_hat.shape[0] == 1, (
            "Internal issue with generation should return single combination."
        )

        return x_hat[0]

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
        x_hat : Array (m, B, W, H, C) padded
            Estimation of clean signal using combinations of auxiliary functions
        phys_sig : Array (B, W, H, C) padded
            Estimation of the known physics signal
        phys_params : list[Array (B, Z)]
            Estimation of physics parameters
        """
        if self.dimension == 2:
            assert x.shape[1:] == (
                self.time_dim + sum(self.pad[0]),
                self.num_grid_points + sum(self.pad[1]),
                1,
            ), (
                f"Unexpected shape for input, {x.shape} != ({x.shape[0]}, {self.time_dim + sum(self.pad[0])}, {self.num_grid_points + sum(self.pad[1])}, 1)"
            )
        elif self.dimension == 1:
            assert x.shape[1:] == (
                self.time_dim + sum(self.pad[0]),
                1,
            ), (
                f"Unexpected shape for input, {x.shape} != ({x.shape[0]}, {self.time_dim + sum(self.pad[0])}, 1)"
            )

        # x_hat : (B, T, X, 1)
        h, aux_latents, phys_params, phys_sig = self._embed(x, tau)

        x_hat = self._generate(
            x,
            h,
            aux_latents,
            phys_params,
            self.pre_combs,
            self.post_combs,
            self.pre_selections,
            self.post_selections,
        )

        assert len(x_hat.shape) == self.dimension + 3, (
            "Internal issue with generation should return combination solution."
        )
        assert x_hat.shape[0] == self.pre_combs * self.post_combs, (
            "Internal issue with generation should return all combinations."
        )

        # x_hat: (m, B, T, X, 1), phys_sig : (B, T, X, 1), phys_params : (B, Z)
        return x_hat, phys_sig, phys_params


class PAID(nnx.Module):
    """
    Physics aligned interpretable diffusion (PAID) model.
    """

    def __init__(
        self,
        unet: dict,
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
        condition: int = 5,
        pad: list[tuple] = [(0, 0), (0, 0)],
        *,
        rngs: nnx.Rngs,
    ) -> None:
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
        self.integral_level = pde["integral_level"]
        self.time_dim = pde["time_dim"]

        self.dt = pde["timestep"] / float(self.integral_level)
        self.t1 = (self.time_dim - 1) * self.dt
        self.t_span = jnp.linspace(0.0, self.t1, self.time_dim)

        self.num_grid_points = pde["num_grid_points"]
        ds = 0.0
        if self.num_grid_points > 1:
            ds = pde["spatial_length"] / (self.num_grid_points - 1)

        physics_model, _, num_derivatives = get_physics_model(pde["system"])
        self.pde_dim = self.num_grid_points * num_derivatives
        self.physics_model = physics_model(jnp.array(ds), "periodic")
        self.solver = get_solver(pde["solver"])

        assert self.physics_model is not None
        assert self.solver is not None

        # Denoiser
        self.denoiser = PhysUNet2D(
            unet,
            pde,
            aux_funcs,
            unmixer,
            phys_funcs,
            pre_funcs,
            post_funcs,
            self.integral_level,
            self.t_span,
            self.t1,
            self.dt,
            self.num_grid_points,
            self.pde_dim,
            self.time_dim,
            self.physics_model,
            self.solver,
            pad,
            rngs,
        )

        # Additional constants
        self.condition = condition
        self.phys_param_ranges = pde["phys_param_ranges"]
        self.pad = pad
        self.dimension = self.denoiser.dimension

    def __call__(self, x: jax.Array, t: jax.Array) -> jax.Array:
        return self.denoiser.forward(x, t)

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
        gen_phys : Array (B, W, H, C)
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

        # ^(B, T, X)
        gen_phys = jnp.reshape(
            # ^(T, B, X*D) => ^(B, T, X)
            jnp.permute_dims(y_seq[:, :, : self.num_grid_points], (1, 0, 2)),
            (-1, self.time_dim, self.num_grid_points),
        )
        if self.dimension == 2:
            # gen_phys : ^(B, T, X, 1)
            gen_phys = jnp.expand_dims(gen_phys, axis=-1)

        return gen_phys
