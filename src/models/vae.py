import jax
import jax.numpy as jnp
from flax import nnx

from models.modules import make_module
from utils.config.models import ModelArch, VAEConfig


class Encoder(nnx.Module):
    def __init__(self, arch: ModelArch, latent_dim=2, *, rngs: nnx.Rngs) -> None:
        self.latent_dim = latent_dim

        self.layers = nnx.List()
        for layer in arch:
            self.layers.append(
                make_module(
                    layer.type, layer.config, out_features=2 * latent_dim, rngs=rngs
                )
            )

    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        assert len(x.shape) == 4, f"Expected B,H,W,C but only received {len(x.shape)}"

        for layer in self.layers:
            x = layer(x)

        mu, log_sigma = jnp.split(x, 2, axis=1)
        sigma = jnp.exp(log_sigma)

        assert mu.shape == (x.shape[0], self.latent_dim)
        assert mu.shape == sigma.shape

        return mu, sigma


class Decoder(nnx.Module):
    def __init__(
        self,
        arch: ModelArch,
        latent_dim: int,
        channel_start: int = 6,
        initial_shape: int = 7,
        target_shape: int = 28,
        eps: float = 1e-6,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.latent_dim = latent_dim
        self.c_start = channel_start
        self.length_start = initial_shape
        self.l_target = target_shape
        self.eps = eps

        self.expand = nnx.Linear(
            latent_dim, initial_shape * initial_shape * channel_start, rngs=rngs
        )
        self.layers = nnx.List()
        for layer in arch:
            self.layers.append(
                make_module(layer.type, layer.config, out_features=1, rngs=rngs)
            )

    def __call__(self, z: jax.Array) -> jax.Array:
        assert len(z.shape) == 2
        assert z.shape[1] == self.latent_dim

        x = self.expand(z).reshape(
            z.shape[0], self.length_start, self.length_start, self.c_start
        )

        for layer in self.layers:
            x = layer(x)

        assert x.shape == (z.shape[0], self.l_target, self.l_target, 1)

        return jnp.clip(x, self.eps, 1 - self.eps)


class VAE(nnx.Module):
    def __init__(self, cfg: VAEConfig, rngs: nnx.Rngs) -> None:
        self.latent_dim = cfg.latent_dim

        self.encode = Encoder(arch=cfg.encoder, latent_dim=cfg.latent_dim, rngs=rngs)
        self.decode = Decoder(
            arch=cfg.decoder,
            latent_dim=cfg.latent_dim,
            channel_start=cfg.channel_start,
            initial_shape=cfg.initial_shape,
            target_shape=cfg.output_shape,
            eps=cfg.eps,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array, key: jax.Array) -> jax.Array:
        sample_z = self.encode_and_sample(x, key)
        return self.decode(sample_z)

    def encode_and_sample(self, x: jax.Array, key: jax.Array) -> jax.Array:
        """
        Encode a given image and sample a latent.

        Parameters
        ----------
        x : Array
            The input image
        key : Array
            PRNG key

        Returns
        -------
        latent : Array
            The sampled latent
        """
        mu_z, sigma_z = self.encode(x)
        Z = jax.random.normal(key=key, shape=mu_z.shape)
        return mu_z + Z * sigma_z
