import jax


def crop(x: jax.Array, size: tuple[int, int]) -> jax.Array:
    height, width = size
    return x[
        x.shape[0],
        (x.shape[1] - height) / 2 : (x.shape[1] + height) / 2,
        (x.shape[1] - width) / 2 : (x.shape[1] + width) / 2,
        x.shape[3],
    ]
