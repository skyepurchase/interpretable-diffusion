from flax import nnx

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
