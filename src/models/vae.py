import distrax
import jax
from jax.lax import cond
from jax.tree_util import Partial
import jax.numpy as jnp
from flax import nnx

from flax.typing import Shape
from jax.image import ResizeMethod


def upsample(x, shape: Shape, method: str | ResizeMethod):
    return jax.image.resize(x, shape=(x.shape[0], *shape), method=method)


class Encoder(nnx.Module):
    def __init__(
        self,
        latent_dim = 2,
        *,
        rngs: nnx.Rngs
    ) -> None:
        self.latent_dim = latent_dim

        self.encode = nnx.Sequential(
            nnx.Conv(1, 16, (3,3), padding='VALID', rngs=rngs),
            nnx.BatchNorm(16, rngs=rngs),
            nnx.relu,
            Partial(nnx.max_pool, window_shape=(3,3)),
            nnx.Conv(16, 32, (3,3), padding='VALID', rngs=rngs),
            nnx.BatchNorm(32, rngs=rngs),
            nnx.relu,
            lambda x: jnp.reshape(x, (x.shape[0], -1)),
            nnx.Linear(32 * 2 * 2, 32, rngs=rngs),
            nnx.Linear(32, 2*latent_dim, rngs=rngs)
        )

    def __call__(self, x) -> distrax.DistributionLike:
        assert len(x.shape) == 4, f"Expected B,H,W,C but only received {len(x.shape)}"

        mu, log_sigma = jnp.split(self.encode(x), 2, axis=1)
        sigma = jnp.exp(log_sigma)

        assert mu.shape == (x.shape[0], self.latent_dim)
        assert mu.shape == sigma.shape

        return distrax.MultivariateNormalDiag(mu, sigma)


class Decoder(nnx.Module):
    def __init__(
        self,
        latent_dim: int,
        channel_start: int = 6,
        length_start: int = 4,
        *,
        rngs: nnx.Rngs
    ) -> None:
        self.latent_dim = latent_dim
        self.c_start = channel_start
        self.length_start = length_start

        self.expand = nnx.Linear(
            self.latent_dim,
            length_start * length_start * channel_start,
            rngs=rngs
        )
        self.generate = nnx.Sequential(
            nnx.Conv(channel_start, 32, (3,3), padding='SAME', rngs=rngs),
            nnx.BatchNorm(32, rngs=rngs),
            nnx.relu,
            Partial(upsample, shape=(7, 7, 32), method="bilinear"),
            nnx.Conv(32, 16, (3,3), padding='VALID', rngs=rngs),
            nnx.BatchNorm(16, rngs=rngs),
            nnx.relu,
            Partial(upsample, shape=(10, 10, 16), method="bilinear"),
            nnx.Conv(16, 1, (3,3), padding='VALID', rngs=rngs),
            nnx.BatchNorm(1, rngs=rngs),
            nnx.sigmoid
        )

    def __call__(self, z) -> distrax.DistributionLike:
        assert len(z.shape) == 2
        assert z.shape[1] == self.latent_dim

        features = self.expand(z).reshape(
            z.shape[0],
            self.length_start,
            self.length_start,
            self.c_start
        )
        mu = self.generate(features)

        assert mu.shape[1] == mu.shape[2], "Expected image to be a square"
        assert mu.shape[1] == 8, f"Expected side length to be 8 received {mu.shape[1]}"

        return distrax.Bernoulli(mu)


class VAE(nnx.Module):
    def __init__(
        self,
        latent_dim: int=2,
        *,
        rngs: nnx.Rngs
    ) -> None:
        self.encoder = Encoder(latent_dim=latent_dim, rngs=rngs)
        self.decoder = Decoder(latent_dim=latent_dim, channel_start=6, rngs=rngs)

    def __call__(self, x, rngs):
        sample_z = self.encode(x).sample(seed=rngs)
        return self.decode(sample_z).sample(seed=rngs)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)
