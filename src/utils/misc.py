from typing import Protocol

import jax
from flax import nnx

Distribution = tuple[jax.Array, jax.Array]


class TrainStep(Protocol):
    def __call__(
        self,
        model: nnx.Module,
        data: jax.Array,
        optimizer: nnx.Optimizer,
        key: jax.Array,
    ) -> jax.Array: ...
