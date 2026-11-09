import math
from typing import Iterator

import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray


class ArrayLoader:
    """Minimal replacement for torch DataLoader over in-memory JAX arrays.

    Yields ``(data, coeffs)`` tuples of batched arrays. When ``shuffle`` is
    True each iteration uses a fresh permutation derived from ``key`` and the
    number of prior iterations, so every epoch is shuffled differently but
    reproducible. The tail partial batch is kept (torch's ``drop_last=False``
    semantics) and ``__len__`` matches the number of batches yielded.
    """

    def __init__(
        self,
        data: Array,
        coeffs: Array,
        batch_size: int,
        *,
        key: PRNGKeyArray,
        shuffle: bool = True,
    ) -> None:
        assert data.shape[0] == coeffs.shape[0], (
            f"Data and coefficients have different batch dimensions, {data.shape[0]} != {coeffs.shape[0]}"
        )
        self.data = data
        self.coeffs = coeffs
        self.batch_size = batch_size
        self.key = key
        self.shuffle = shuffle
        self._iterations = 0

    def __len__(self) -> int:
        return math.ceil(self.data.shape[0] / self.batch_size)

    def __iter__(self) -> Iterator[tuple[Array, Array]]:
        n = self.data.shape[0]
        if self.shuffle:
            key = jax.random.fold_in(self.key, self._iterations)
            self._iterations += 1
            order = jax.random.permutation(key, n)
        else:
            order = jnp.arange(n)

        for start in range(0, n, self.batch_size):
            idx = order[start : start + self.batch_size]
            if len(idx) < self.batch_size:
                # Ensure every batch is the same size
                pad = self.batch_size - len(idx)
                idx = jnp.concatenate([idx, order[:pad]], axis=0)
            yield self.data[idx], self.coeffs[idx]
