from functools import reduce

import jax
import jax.numpy as jnp
from diffuse.neural_network.blocks import ResnetBlock
from diffuse.neural_network.nn import CondUNet2D
from flax import nnx

from models.ddim import compute_betas
from models.modules.acts import ACTS
from utils.config.models import PAIDConfig


class PhysUNet2D(nnx.Module):
    def __init__(self, cfg: PAIDConfig, rngs: nnx.Rngs):
        self.unet = CondUNet2D(
            cfg.in_features,
            cfg.base_features,
            cfg.feature_multiples,
            cfg.num_blocks,
            cfg.attention_resolutions,
            ACTS[cfg.act],
            cfg.dropout,
            cfg.dropout_rate,
            cfg.attention_heads,
            jnp.float32,
            jnp.float32,
            rngs,
        )

        aux_dims = []
        latent_features = cfg.feature_multiples[-1] * cfg.base_features
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

        out_features = cfg.base_features + reduce(lambda x, y: x + y, aux_dims)
        self.unet.out = nnx.Sequential(
            nnx.GroupNorm(
                num_features=out_features,
                num_groups=out_features,
                epsilon=1e-6,
                param_dtype=jnp.float32,
                dtype=jnp.float32,
                rngs=rngs,
            ),
            ACTS[cfg.act],
            nnx.Conv(
                in_features=out_features,
                out_features=cfg.in_features,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding=(1, 1),
                param_dtype=jnp.float32,
                dtype=jnp.float32,
                rngs=rngs,
            ),
        )

        self.unmixer = nnx.Sequential(nnx.identity)
        self.phys_funcs = nnx.List()

        self.post_funcs = nnx.List()

    def forward(self, x: jax.Array, t: jax.Array) -> jax.Array:
        # x : (B, X, T, 1)
        h = x

        t_emb = self.unet.time_proj(t)
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

        # phys : (B, X, T, 1 + z*n)
        phys = self.unmixer(jnp.concatenate([x] + aux_latents, axis=3))

        # phys_latent : (B, Z)
        phys_latents = []
        for phys_func in self.phys_funcs:
            phys_latents.append(phys_func(phys))

        for block in self.unet.blocks_up:
            h = jnp.concatenate([h, hs.pop()], axis=-1)
            h = block(h, t_emb)
        # h : (B, X, T, 1)

        # Solve ODE
        # y_seq : (S, B, X*D)
        # phys_h : (B, X, T, 1)
        phys_h = h

        for z_aux, post_func in zip(aux_latents, self.post_funcs):
            h = jnp.concatenate([h, phys_h], axis=3)
            h = post_func(h, z_aux)

        # h : (B, X, T, 1)
        h = jnp.concatenate([h] + aux_latents, axis=3)
        return self.unet.out(h)


class PAID(nnx.Module):
    """
    Physics aligned interpretable diffusion (PAID) model.
    """

    def __init__(self, cfg: PAIDConfig, rngs: nnx.Rngs) -> None:
        self.num_timesteps = cfg.num_timesteps
        self.betas = compute_betas(
            cfg.beta_schedule, cfg.beta_start, cfg.beta_end, self.num_timesteps
        )

        self.denoiser = PhysUNet2D(cfg, rngs)

        self.pad = cfg.pad

    def __call__(self, x: jax.Array, t: jax.Array) -> jax.Array:
        return self.denoiser.forward(x, t)
