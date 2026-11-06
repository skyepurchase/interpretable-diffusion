import jax.numpy as jnp
from jax import Array


def remove_padding(a: Array, padding: list[tuple[int, int]]):
    """
    Remove padding from images only when padding is positive for that dimension.

    Parameters
    a : ArrayLike (B, ..., C)
        Array to remove padding from. Assumes it is batched with a channel.
    padding : list[tuple[int, int]]
        The padding added to the array.

    Returns
    -------
    a : Arraylike (B, ..., C)
        Array without the padding
    """
    assert len(padding) == len(a.shape) - 2, (
        f"Number of padding dimensions needs to match a padding dimensions, {len(padding)} != {len(a.shape) - 2}."
    )
    assert all(
        map(lambda p: len(p) == 2, padding)
    ), f"Expected padding to specify two numbers, received {
        list(map(lambda p: len(p), padding))
    }."

    for dim in range(1, len(a.shape) - 1):
        length = a.shape[dim]
        left_pad = padding[dim - 1][0]
        right_pad = padding[dim - 1][1]

        assert length > left_pad + right_pad, (
            f"Unable to remove padding from axis {dim}, {length} <= {left_pad} + {right_pad}."
        )

        _, a, _ = jnp.split(a, [left_pad, length - right_pad], axis=dim)

    return a


TIME_ENCODINGS = ("raw", "sin_cos", "both")

TIME_ENCODING_DIMS = {"raw": 1, "sin_cos": 2, "both": 3}


def time_features(t, template: Array, encoding: str) -> Array:
    """
    Encode the ODE time as features for the in-ODE auxiliary functions.

    Parameters
    ----------
    t : RealScalarLike
        The current time
    template : Array (B, D)
        Batched array whose first column shapes the time vector
    encoding : str
        One of "raw" (t), "sin_cos" (sin t, cos t), or "both"
        (t, sin t, cos t)

    Returns
    -------
    features : Array (B, 1 | 2 | 3)
        The time features, matching TIME_ENCODING_DIMS[encoding]
    """
    t_vec = jnp.ones_like(template)[:, 0, None] * t
    if encoding == "raw":
        return t_vec
    if encoding == "sin_cos":
        return jnp.concatenate([jnp.sin(t_vec), jnp.cos(t_vec)], axis=1)
    if encoding == "both":
        return jnp.concatenate([t_vec, jnp.sin(t_vec), jnp.cos(t_vec)], axis=1)
    raise ValueError(f"Unknown time encoding: {encoding}")
