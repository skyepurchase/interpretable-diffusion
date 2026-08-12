from flax import nnx
from jaxtyping import Array

from models.modules.acts import ACTS


def make_mlp(
    in_features: int,
    hidden_features: list[int],
    out_features: int,
    act: str,
    rngs: nnx.Rngs,
) -> nnx.Module:
    """Create a simple multilayer perceptron according to the given configuration."""
    dims = [in_features] + hidden_features

    return nnx.Sequential(
        *[
            nnx.Sequential(nnx.Linear(in_features, out_features, rngs=rngs), ACTS[act])
            for in_features, out_features in zip(dims[:-1], dims[1:])
        ],
        # Assume distribution is normal
        nnx.Linear(dims[-1], out_features, rngs=rngs),
    )


class Mlp(nnx.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: list[int],
        out_features: int,
        act: str,
        rngs: nnx.Rngs,
    ) -> None:
        dims = [in_features] + hidden_features
        activation = ACTS[act]

        self.layers = nnx.Sequential(
            *[
                nnx.Sequential(
                    nnx.Linear(in_features, out_features, rngs=rngs), activation
                )
                for in_features, out_features in zip(dims[:-1], dims[1:])
            ],
            nnx.Linear(hidden_features[-1], out_features, rngs=rngs),
        )

    def __call__(self, x: Array) -> Array:
        return self.layers(x)
