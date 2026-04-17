from functools import reduce

import jax
import jax.numpy as jnp
from diffrax import ODETerm, SaveAt, diffeqsolve
from diffrax._custom_types import RealScalarLike
from diffuse.neural_network.blocks import ResnetBlock
from diffuse.neural_network.nn import CondUNet2D
from flax import nnx
from jaxtyping import PyTree

from models.ddim import compute_betas
from models.mlp import make_mlp
from models.modules.acts import ACTS
from models.phys_vae import NUM_DERIVATIVES, PHYS_MODELS, SOLVERS, gen_selections
from models.utils import remove_padding
from utils.config.models import PAIDConfig


class PhysUNet2D(nnx.Module):
    """TODO: Add docstring"""

    def __init__(self, cfg: PAIDConfig, pad: list[tuple[int, int]], rngs: nnx.Rngs):
        # Constants
        self.pad = pad

        num_pre_funcs = len(cfg.pre_funcs)
        num_post_funcs = len(cfg.post_funcs)
        self.pre_combs = 2**num_pre_funcs
        self.post_combs = 2**num_post_funcs
        self.pre_selections, self.post_selections = gen_selections(
            num_pre_funcs, num_post_funcs
        )

        ## PDE solver constants
        self.integral_level = cfg.pde.integral_level
        self.integral_len = (cfg.pde.time_dim - 1) * self.integral_level + 1
        self.integral_step = cfg.pde.timestep / float(self.integral_level)
        self.t_span = jnp.linspace(
            0.0, self.integral_step * (self.integral_len - 1), self.integral_len
        )

        self.t1 = self.integral_len
        self.dt = self.integral_step
        self.num_grid_points = cfg.pde.num_grid_points
        self.pde_dim = self.num_grid_points * NUM_DERIVATIVES.get(cfg.pde.system, 1)
        self.time_dim = cfg.pde.time_dim

        if self.num_grid_points > 1:
            ds = cfg.pde.length / (self.num_grid_points - 1)
        else:
            ds = 0.0

        # PDE solver functions
        if cfg.pde.system not in PHYS_MODELS:
            raise NotImplementedError(
                f"Physics model {cfg.pde.system} is not implemented."
            )
        self.physics_model = PHYS_MODELS.get(cfg.pde.system, lambda _: None)(
            jnp.array(ds, dtype=jnp.float32)
        )

        if cfg.pde.solver not in SOLVERS:
            raise NotImplementedError(f"Solver {cfg.pde.solver} is not implemented.")
        self.solver = SOLVERS.get(cfg.pde.solver)

        # Modules
        # temporal Unet
        self.unet = CondUNet2D(
            cfg.unet.in_features,
            cfg.unet.base_features,
            cfg.unet.feature_multiples,
            cfg.unet.num_blocks,
            cfg.unet.attention_resolutions,
            ACTS[cfg.unet.act],
            cfg.unet.dropout,
            cfg.unet.dropout_rate,
            cfg.unet.attention_heads,
            jnp.float32,
            jnp.float32,
            rngs,
        )
        unet_latent_features = cfg.unet.feature_multiples[-1] * cfg.unet.base_features

        # Auxiliary functions
        aux_dims = []
        self.aux_funcs = nnx.List()
        for aux_cfg in cfg.aux_funcs:
            aux_func = nnx.Sequential(
                ResnetBlock(
                    unet_latent_features,
                    aux_cfg.features,
                    ACTS[aux_cfg.act],
                    embedding_dim=aux_cfg.time_dim,
                    dropout=aux_cfg.dropout,
                    dropout_rate=aux_cfg.dropout_rate,
                    rngs=rngs,
                ),
                ACTS[aux_cfg.act],
                lambda x: x.reshape(x.shape[0], -1),
                nnx.Linear(aux_cfg.in_features, aux_cfg.latent_dim, rngs=rngs),
            )
            self.aux_funcs.append(aux_func)
            aux_dims.append(aux_cfg.latent_dim)

        # Known physics signal estimators
        latent_dims = reduce(lambda x, y: x + y, aux_dims)
        self.unmixer_conv = nnx.Conv(
            in_features=latent_dims + 1,
            out_features=cfg.unmixer.dims[0],
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=(1, 1),
            rngs=rngs,
        )
        self.unmixer = nnx.List([])
        for in_features, out_features in zip(
            cfg.unmixer.dims[:-1], cfg.unmixer.dims[1:]
        ):
            self.unmixer.append(
                ResnetBlock(
                    in_channels=in_features,
                    out_channels=out_features,
                    activation=ACTS[cfg.unmixer.act],
                    embedding_dim=cfg.unmixer.time_dim,
                    dropout=cfg.unmixer.dropout,
                    dropout_rate=cfg.unmixer.dropout_rate,
                    rngs=rngs,
                )
            )
            in_features = out_features
        self.unmixer_out = nnx.Conv(
            in_features=cfg.unmixer.dims[-1],
            out_features=1,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=(1, 1),
            rngs=rngs,
        )

        # Physic parameter functions
        assert len(cfg.phys_funcs) == len(cfg.pde.phys_param_ranges), (
            f"Number of physics model parameters must match number of parameter estimators, {len(cfg.pde.phys_param_ranges)} != {len(cfg.phys_funcs)}"
        )
        self.phys_funcs = nnx.List()
        for phys_cfg, param_range in zip(cfg.phys_funcs, cfg.pde.phys_param_ranges):
            phys_func = nnx.Sequential(
                ResnetBlock(
                    unet_latent_features,
                    phys_cfg.features,
                    ACTS[phys_cfg.act],
                    embedding_dim=phys_cfg.time_dim,
                    dropout=phys_cfg.dropout,
                    dropout_rate=phys_cfg.dropout_rate,
                    rngs=rngs,
                ),
                ACTS[phys_cfg.act],
                lambda x: x.reshape(x.shape[0], -1),
                nnx.Linear(phys_cfg.in_features, phys_cfg.latent_dim, rngs=rngs),
                nnx.sigmoid,
                lambda x: x
                * ((2 * param_range[1]) - (param_range[0] / 2) + (param_range[0] / 2)),
            )
            self.phys_funcs.append(phys_func)

        # Perturbations
        ## PDE perturbations
        self.pre_funcs = nnx.List()
        for func_cfg in cfg.pre_funcs:
            # Z + X*D + t -> dy
            func = make_mlp(
                func_cfg.latent_dim + self.pde_dim + 1,
                func_cfg.dims,
                self.num_grid_points,
                func_cfg.act,
                rngs=rngs,
            )
            self.pre_funcs.append(func)

        ## Post PDE perturbations
        self.post_funcs = nnx.List()
        for func_cfg in cfg.post_funcs:
            func = nnx.Sequential(
                nnx.Conv(
                    in_features=2 + func_cfg.latent_dim,
                    out_features=func_cfg.features,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding=(1, 1),
                    rngs=rngs,
                ),
                nnx.GroupNorm(
                    num_features=func_cfg.features,
                    num_groups=32,
                    rngs=rngs,
                ),
                nnx.Conv(
                    in_features=func_cfg.features,
                    out_features=1,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding=(1, 1),
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
                        jnp.expand_dims(full_aux, axis=[1, 2]),
                        [1, x.shape[1], x.shape[2], 1],
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
        x_hat, aux_latents, phys_latents, _ = self._embed(x, tau)

        def PDE(t: RealScalarLike, y: PyTree, _args: PyTree):
            """Internal PDE to be solved during the generation process."""
            # y : (B, X*D)
            assert len(y.shape) == 2, f"expected (B, X*D) but received {y.shape}"
            assert y.shape[1] == self.pde_dim, (
                f"Expected (B, {self.pde_dim}) but received {y.shape}"
            )

            # y_dot_phys : (B, X*D)
            y_dot_phys = self.physics_model(y, *phys_latents)

            # y_dot_aux : (B, X*D)
            y_dot_aux = jnp.zeros_like(y_dot_phys)
            for z_aux, pre_func in zip(aux_latents, self.pre_funcs):
                y_dot_aux += pre_func(
                    jnp.concatenate(
                        [z_aux, y, jnp.ones_like(y)[:, 0, None] * t],
                        axis=-1,
                    )
                )

            # ys_dot: (B, X*D)
            return y_dot_phys + y_dot_aux

        # init_cond : (B, X*D)
        init_cond = jnp.pad(
            remove_padding(x, self.pad)[:, 0, :, 0],
            ((0, 0), (0, self.pde_dim - self.num_grid_points)),
        )

        # y_seq : (S, B, X*D)
        y_seq = diffeqsolve(
            ODETerm(PDE),
            self.solver(),
            t0=0.0,
            t1=self.t1,
            dt0=self.dt,
            y0=init_cond,
            saveat=SaveAt(ts=self.t_span),
        ).ys
        assert y_seq is not None, "Failed to solve PDE"

        # trim_y_seq : (T, B, X*D)
        trim_y_seq = y_seq[jnp.arange(0, self.t1, self.integral_level)]

        # gen_phys : (B, T, X, 1)
        gen_phys = jnp.expand_dims(
            # (T, B, X*D) => (T, B, X) => (B, T, X)
            jnp.permute_dims(trim_y_seq[:, :, : self.num_grid_points], (1, 0, 2)),
            axis=3,
        )
        gen_phys = jnp.pad(gen_phys, [(0, 0), self.pad[0], self.pad[1], (0, 0)])

        for z_aux, post_func in zip(
            aux_latents[len(self.pre_funcs) :], self.post_funcs
        ):
            # z_expanded : (B, T, X, Z)
            z_expanded = jnp.tile(
                jnp.expand_dims(z_aux, axis=[1, 2]),
                [1, x_hat.shape[1], x_hat.shape[2], 1],
            )
            # h : (B, T, X, 1)
            x_hat = post_func(jnp.concatenate([x_hat, gen_phys, z_expanded], axis=3))

        # x_hat : (B, T, X, 1)
        return x_hat

    def full_forward(
        self, x: jax.Array, tau: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
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
        phys_sig : Array (B, W, H, C) padded
            Estimation of the known physics signal
        gen_phys : Array (B, W, H, C) padded
            Solved physics
        x_hat : Array (m, B, W, H, C) padded
            Estimation of clean signal, should be close to gen_phys
        """
        # x_hat : (B, T, X, 1)
        h, aux_latents, phys_latents, phys_sig = self._embed(x, tau)

        def PDE(t: RealScalarLike, y: PyTree, _args: PyTree):
            """Internal PDE to be solved during the generation process."""
            # y : (B*n, X*D)
            assert len(y.shape) == 2, f"expected (B, X*D) but received {y.shape}"
            assert y.shape[1] == self.pde_dim, (
                f"Expected (B, {self.pde_dim}) but received {y.shape}"
            )

            ys = y.reshape(self.pre_combs, -1, self.pde_dim)

            # y_dot_phys : (n, B, X*D)
            ys_dot_phys = jax.vmap(self.physics_model, in_axes=(0, None), out_axes=0)(
                ys, *phys_latents
            )

            # ys_dot_aux : (n, B, X*D)
            ys_dot_aux = jnp.zeros_like(ys_dot_phys)
            for z_aux, pre_func, selections in zip(
                aux_latents, self.pre_funcs, self.pre_selections
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
                remove_padding(x, self.pad)[:, 0, :, 0],
                ((0, 0), (0, self.pde_dim - self.num_grid_points)),
            ),
            reps=(self.pre_combs, 1),
        )

        # y_seq : ^(S, B*n, X*D)
        y_seq = diffeqsolve(
            ODETerm(PDE),
            self.solver(),
            t0=0.0,
            t1=self.t1,
            dt0=self.dt,
            y0=init_cond,
            saveat=SaveAt(ts=self.t_span),
        ).ys
        assert y_seq is not None, "Failed to solve PDE"

        # trim_y_seq : ^(T, B*n, X*D)
        trim_y_seq = y_seq[jnp.arange(0, self.t1, self.integral_level)]

        # gen_phys : ^(n, B, T, X, 1)
        x_hat = jnp.expand_dims(
            # ^(n, B, T, X)
            jnp.reshape(
                # ^(T, B*n, X*D) => ^(B*n, T, X)
                jnp.permute_dims(trim_y_seq[:, :, : self.num_grid_points], (1, 0, 2)),
                (self.pre_combs, -1, self.time_dim, self.num_grid_points),
            ),
            axis=-1,
        )
        # gen_phys : (m, B, T, X, 1)
        x_hat = jnp.pad(
            jnp.tile(x_hat, [self.post_combs, 1, 1, 1, 1]),
            [(0, 0), (0, 0), self.pad[0], self.pad[1], (0, 0)],
        )

        for z_aux, post_func, selections in zip(
            aux_latents[len(self.pre_funcs) : len(self.post_funcs)],
            self.post_funcs,
            self.post_selections,
        ):
            # z_expanded : (B, T, X, Z)
            z_expanded = jnp.tile(
                jnp.expand_dims(z_aux, axis=[1, 2]),
                [1, h.shape[1], h.shape[2], 1],
            )

            def apply_func(phys, apply):
                return (1 - apply) * phys + apply * post_func(
                    jnp.concatenate([h, phys, z_expanded], axis=3)
                )

            # x_hat : (m, B, T, X, 1)
            x_hat = jax.vmap(apply_func, in_axes=(0, 0), out_axes=0)(x_hat, selections)

        # phys_sig : (B, T, X, 1), gen_phys: (B, T, X, 1), h : (m, B, T, X, 1)
        return phys_sig, x_hat[0], x_hat


class PAID(nnx.Module):
    """
    Physics aligned interpretable diffusion (PAID) model.
    """

    def __init__(self, cfg: PAIDConfig, rngs: nnx.Rngs) -> None:
        self.num_timesteps = cfg.num_timesteps
        self.betas = compute_betas(
            cfg.beta_schedule, cfg.beta_start, cfg.beta_end, self.num_timesteps
        )

        self.denoiser = PhysUNet2D(cfg, cfg.pad, rngs)

        self.pad = cfg.pad

    def __call__(self, x: jax.Array, t: jax.Array) -> jax.Array:
        return self.denoiser.forward(x, t)
