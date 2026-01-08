import jax
from jax.tree_util import Partial
import jax.numpy as jnp
from flax import nnx

from flax.typing import Shape
from jax.image import ResizeMethod


class Upsample:
    def __init__(
        self,
        shape: Shape,
        method: str | ResizeMethod = "bilinear"
    ) -> None:
        self.shape = shape
        self.method = method

    def __call__(self, x: jax.Array) -> jax.Array:
        assert len(x.shape) == 4, f"Expected image to have shape B,W,H,C received {len(x.shape)} dims"

        batch = x.shape[0]
        channels = x.shape[-1]
        return jax.image.resize(
            x,
            shape=(batch, *self.shape, channels),
            method=self.method
        )


class Encoder(nnx.Module):
    def __init__(
        self,
        latent_dim = 2,
        *,
        rngs: nnx.Rngs
    ) -> None:
        self.latent_dim = latent_dim

        self.encode = nnx.Sequential(
            nnx.Conv(1, 16, (3,3), strides=(3,3), padding='VALID', rngs=rngs),
            nnx.BatchNorm(16, rngs=rngs),
            nnx.relu,
            Partial(nnx.max_pool, window_shape=(2,2)),
            nnx.Conv(16, 32, (3,3), strides=(3,3), padding='VALID', rngs=rngs),
            nnx.BatchNorm(32, rngs=rngs),
            nnx.relu,
            lambda x: jnp.reshape(x, (x.shape[0], -1)),
            nnx.Linear(32 * 2 * 2, 32, rngs=rngs),
            nnx.Linear(32, 2*latent_dim, rngs=rngs)
        )

    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        assert len(x.shape) == 4, f"Expected B,H,W,C but only received {len(x.shape)}"

        mu, log_sigma = jnp.split(self.encode(x), 2, axis=1)
        sigma = jnp.exp(log_sigma)

        assert mu.shape == (x.shape[0], self.latent_dim)
        assert mu.shape == sigma.shape

        return mu, sigma


class Decoder(nnx.Module):
    def __init__(
        self,
        latent_dim: int,
        channel_start: int = 6,
        length_start: int = 7,
        target_length: int = 28,
        eps: float = 1e-6,
        *,
        rngs: nnx.Rngs
    ) -> None:
        self.latent_dim = latent_dim
        self.c_start = channel_start
        self.length_start = length_start
        self.l_target = target_length
        self.eps = eps

        self.expand = nnx.Linear(
            self.latent_dim,
            length_start * length_start * channel_start,
            rngs=rngs
        )
        self.generate = nnx.Sequential(
            nnx.Conv(channel_start, 32, (3,3), padding='SAME', rngs=rngs),
            nnx.BatchNorm(32, rngs=rngs),
            nnx.relu,
            Upsample((15,15), method="bilinear"),
            nnx.Conv(32, 16, (3,3), padding='VALID', rngs=rngs),
            nnx.BatchNorm(16, rngs=rngs),
            nnx.relu,
            Upsample((30,30), method="bilinear"),
            nnx.Conv(16, 1, (3,3), padding='VALID', rngs=rngs),
            nnx.BatchNorm(1, rngs=rngs)
        )

    def __call__(self, z: jax.Array) -> jax.Array:
        assert len(z.shape) == 2
        assert z.shape[1] == self.latent_dim

        features = self.expand(z).reshape(
            z.shape[0],
            self.length_start,
            self.length_start,
            self.c_start
        )
        logits = self.generate(features)

        assert logits.shape[1] == logits.shape[2], "Expected image to be a square"
        assert logits.shape[1] == self.l_target, f"Expected side length to be {self.l_target} received {logits.shape[1]}"

        return jnp.clip(
            nnx.sigmoid(logits), self.eps, 1-self.eps)


class VAE(nnx.Module):
    def __init__(
        self,
        latent_dim: int=2,
        channel_start: int = 6,
        length_start: int = 7,
        target_length: int = 28,
        eps: float = 1e-6,
        *,
        rngs: nnx.Rngs
    ) -> None:
        self.latent_dim = latent_dim

        self.encode = Encoder(
            latent_dim=latent_dim, rngs=rngs)
        self.decode = Decoder(
            latent_dim=latent_dim,
            channel_start=channel_start,
            length_start=length_start,
            target_length=target_length,
            eps=eps,
            rngs=rngs
        )

    def __call__(
        self,
        x: jax.Array,
        key: jax.Array
    ) -> jax.Array:
        sample_z = self.encode_and_sample(x, key)
        return self.decode(sample_z)

    def encode_and_sample(
        self,
        x: jax.Array,
        key: jax.Array
    ) -> jax.Array:
        mu_z, sigma_z = self.encode(x)
        Z = jax.random.normal(key=key, shape=mu_z.shape)
        return mu_z + Z * sigma_z
