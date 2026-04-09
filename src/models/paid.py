from functools import reduce

import jax
import jax.numpy as jnp
from diffrax import ODETerm, SaveAt, diffeqsolve
from diffrax._custom_types import RealScalarLike
from diffuse.neural_network.blocks import Downsample, ResnetBlock
from diffuse.neural_network.nn import CondUNet2D
from flax import nnx
from jaxtyping import PyTree

from models.ddim import compute_betas
from models.modules.acts import ACTS
from models.phys_vae import NUM_DERIVATIVES, PHYS_MODELS, SOLVERS
from utils.config.models import PAIDConfig


class PhysUNet2D(nnx.Module):
    """TODO: Add docstring"""

    def __init__(self, cfg: PAIDConfig, pad: list[tuple[int, int]], rngs: nnx.Rngs):
        # Constants
        self.pad = pad

        # PDE solver constants
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

        # Auxiliary functions
        aux_dims = []
        latent_features = cfg.unet.feature_multiples[-1] * cfg.unet.base_features
        self.aux_funcs = nnx.List()
        for aux_cfg in cfg.aux_funcs:
            aux_func = nnx.Sequential(
                ResnetBlock(
                    latent_features,
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

        # Physics parameter estimators
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
        self.unmixer.append(
            nnx.Conv(
                in_features=cfg.unmixer.dims[-1],
                out_features=1,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding=(1, 1),
                rngs=rngs,
            )
        )

        self.phys_features_conv = nnx.Conv(
            in_features=1,
            out_features=cfg.unet.base_features,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=(1, 1),
            rngs=rngs,
        )
        self.phys_feature_extract = nnx.List([])
        curr_features = cfg.unet.base_features
        for level, mult in enumerate(cfg.unet.feature_multiples):
            for _ in range(cfg.unet.num_blocks):
                self.phys_feature_extract.append(
                    ResnetBlock(
                        in_channels=curr_features,
                        out_channels=int(mult * cfg.unet.base_features),
                        activation=ACTS[cfg.unet.act],
                        dropout=cfg.unet.dropout,
                        dropout_rate=cfg.unet.dropout_rate,
                        rngs=rngs,
                    )
                )

                curr_features = int(mult * cfg.unet.base_features)

            if level != len(cfg.unet.feature_multiples) - 1:
                self.phys_feature_extract.append(
                    Downsample(in_channels=curr_features, rngs=rngs)
                )

        self.phys_funcs = nnx.List()
        assert len(cfg.phys_funcs) == len(cfg.pde.phys_param_ranges), (
            f"Number of physic latent functions and parameter ranges must match, {len(cfg.phys_funcs)} != {len(cfg.pde.phys_param_ranges)}."
        )
        for phys_cfg, param_range in zip(cfg.phys_funcs, cfg.pde.phys_param_ranges):
            phys_func = nnx.Sequential(
                ResnetBlock(
                    curr_features,
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
        self.post_funcs = nnx.List()

        # TODO: REMOVE
        self.conv_merge = nnx.Conv(
            in_features=cfg.unet.base_features + 1,
            out_features=cfg.unet.base_features,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=(1, 1),
            rngs=rngs,
        )

    def forward(self, x: jax.Array, tau: jax.Array) -> jax.Array:
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
            aux_latents.append(
                jnp.tile(
                    jnp.expand_dims(aux_func(h), axis=[1, 2]),
                    [1, x.shape[1], x.shape[2], 1],
                )
            )

        # phys : (B, T, X, 1)
        phys = self.unmixer_conv(jnp.concatenate([x] + aux_latents, axis=3))
        for layer in self.unmixer:
            phys = layer(phys)

        phys_features = self.phys_features_conv(phys)
        for layer in self.phys_feature_extract:
            phys_features = layer(phys_features)

        # phys_latent : (B, Z)
        phys_latents = []
        for phys_func in self.phys_funcs:
            phys_latents.append(phys_func(phys_features))

        # h : (B, T, X, 32)
        for block in self.unet.blocks_up:
            h = jnp.concatenate([h, hs.pop()], axis=-1)
            h = block(h, t_emb)

        def PDE(t: RealScalarLike, y: PyTree, _args: PyTree):
            """Internal PDE to be solved during the generation process."""
            # y : (B, X*D)
            assert len(y.shape) == 2, f"expected (B, X*D) but received {y.shape}"
            assert y.shape[1] == self.pde_dim, (
                f"Expected (B, {self.pde_dim}) but received {y.shape}"
            )

            # y_dot_phys : (B, X*D)
            y_dot_phys = self.physics_model(y, *phys_latents)

            # TODO: Implement auxiliary perturbation
            # y_dot_aux : (B, X*D)
            y_dot_aux = jnp.zeros_like(y_dot_phys)

            # ys_dot: (B, X*D)
            return y_dot_phys + y_dot_aux

        # init_cond : (B, X*D)
        init_cond = jnp.pad(
            x[:, self.pad[0][0], :, 0],
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

        # phys_h : (B, T, X, 1)
        phys_h = jnp.expand_dims(
            # (T, B, X)
            jnp.permute_dims(trim_y_seq[:, :, : self.num_grid_points], (1, 0, 2)),
            axis=3,
        )
        phys_h = jnp.pad(phys_h, [(0, 0), self.pad[0], self.pad[1], (0, 0)])

        for z_aux, post_func in zip(aux_latents, self.post_funcs):
            h = post_func(jnp.concatenate([h, phys_h, z_aux], axis=3))

        h = self.conv_merge(jnp.concatenate([h, phys_h], axis=3))
        # out : (B, T, X, 1)
        return self.unet.out(h)

    def full_forward(self, x: jax.Array, t: jax.Array) -> jax.Array:
        # TODO: Implement
        raise NotImplementedError


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
