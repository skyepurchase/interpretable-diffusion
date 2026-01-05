from flax import nnx
from jax.lax import cond


class Encoder(nnx.Module):
    def __init__(
        self,
        latent_dim = 2,
        *,
        rngs: nnx.Rngs
    ) -> None:
        self.latent_dim = latent_dim

        self.layer1 = nnx.Conv(1, 16, (3,3), rngs=rngs)
        self.layer2 = nnx.Conv(16, 8, (3,3), rngs=rngs)
        self.layer2 = nnx.Conv(8, 16, (4,4), rngs=rngs)

        self.stats = nnx.Linear(16, 2*latent_dim, rngs=rngs)

    def __call__(self, x):
        x = cond(
            len(x.shape) == 3,
            x.reshape(*x.shape, 1), x
        )


class Decoder(nnx.Module):
    def __init__(self) -> None:
        super().__init__()


class VAE(nnx.Module):
    def __init__(
        self,
        n_features: int = 64,
        n_hidden: int = 100,
        n_targets: int = 10,
        *,
        rngs: nnx.Rngs
    ) -> None:
        self.n_features = n_features

        self.layer1 = nnx.Linear(n_features, n_hidden, rngs=rngs)
        self.layer2 = nnx.Linear(n_hidden, n_hidden, rngs=rngs)
        self.layer3 = nnx.Linear(n_hidden, n_targets, rngs=rngs)

    def __call__(self, x):
        x = x.reshape(x.shape[0], self.n_features)
        x = nnx.selu(self.layer1(x))
        x = nnx.selu(self.layer2(x))
        x = self.layer3(x)
        return x
