import jax
import jax.numpy as jnp
from diffrax import ODETerm, SaveAt, diffeqsolve
from diffrax._custom_types import RealScalarLike
from diffuse.neural_network.blocks import ResnetBlock
from diffuse.neural_network.nn import CondUNet2D
from flax import nnx
from jaxtyping import PyTree

from models.ddim import compute_betas
from models.modules.acts import ACTS
from models.phys_vae import NUM_DERIVATIVES, PHYS_MODELS, SOLVERS
from utils.config.models import PAIDConfig


class PhysUNet2D(nnx.Module):
    def __init__(self, cfg: PAIDConfig, rngs: nnx.Rngs):
        # Constants
        # PDE solver constants
        self.integral_level = cfg.pde.integral_level
        self.integral_len = (cfg.pde.time_dim - 1) * self.integral_level + 1
        self.integral_step = cfg.pde.timestep / float(self.integral_level)
        self.t_span = jnp.linspace(
            0.0, self.integral_step * (self.integral_len - 1), self.integral_len
        )

        self.t1 = self.integral_level
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
        self.unmixer = nnx.Sequential(nnx.identity)
        self.phys_funcs = nnx.List()

        # Perturbations
        self.post_funcs = nnx.List()

    def forward(self, x: jax.Array, t: jax.Array) -> jax.Array:
        # x : (B, T, X, 1)
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

        # phys : (B, T, X, 1 + z*n)
        phys = self.unmixer(jnp.concatenate([x] + aux_latents, axis=3))

        # phys_latent : (B, Z)
        phys_latents = []
        for phys_func in self.phys_funcs:
            phys_latents.append(phys_func(phys))

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
            y_dot_phys = self.physics_model(y, 0.05)

            # TODO: Implement auxiliary perturbation
            # y_dot_aux : (B, X*D)
            y_dot_aux = jnp.zeros_like(y_dot_phys)

            # ys_dot: (B, X*D)
            return y_dot_phys + y_dot_aux

        # init_cond : (B, X*D)
        init_cond = jnp.pad(
            x[:, 0, :, 0], ((0, 0), (0, self.pde_dim - self.num_grid_points))
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
            jnp.permute_dims(trim_y_seq[:, :, : self.num_grid_points], (1, 2, 0)),
            axis=3,
        )

        for z_aux, post_func in zip(aux_latents, self.post_funcs):
            h = post_func(jnp.concatenate([h, phys_h, z_aux], axis=3))

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

        self.denoiser = PhysUNet2D(cfg, rngs)

        self.pad = cfg.pad

    def __call__(self, x: jax.Array, t: jax.Array) -> jax.Array:
        return self.denoiser.forward(x, t)
