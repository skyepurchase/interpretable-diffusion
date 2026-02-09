import jax
import jax.numpy as jnp
from flax import nnx

from models.modules.acts import ACTS
from utils.config.models import TUNetConfig


class TResidualBlock(nnx.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        time_features: int,
        act,
        rngs: nnx.Rngs,
    ) -> None:
        super().__init__()
        self.conv1 = nnx.Sequential(
            nnx.Conv(
                in_features=in_features,
                out_features=out_features,
                kernel_size=(3, 3),
                padding="same",
                rngs=rngs,
            ),
            nnx.BatchNorm(out_features, rngs=rngs),
            ACTS[act],
        )
        self.time_proj = nnx.Linear(
            in_features=time_features, out_features=out_features, rngs=rngs
        )
        self.conv2 = nnx.Sequential(
            nnx.Conv(
                in_features=out_features,
                out_features=out_features,
                kernel_size=(3, 3),
                padding="same",
                rngs=rngs,
            ),
            nnx.BatchNorm(out_features, rngs=rngs),
            ACTS[act],
        )

        self.residual = nnx.Conv(
            in_features=in_features,
            out_features=out_features,
            kernel_size=(1, 1),
            strides=(1, 1),
            rngs=rngs,
        )
        self.out_act = ACTS[act]

    def __call__(self, x: jax.Array, t: jax.Array) -> jax.Array:
        h = x
        h = self.conv1(h)
        h += self.time_proj(t)[:, None, None, :]
        h = self.conv2(h)

        x = self.residual(x)

        return self.out_act(h + x)


class TUNet(nnx.Module):
    def __init__(self, cfg: TUNetConfig, rngs: nnx.Rngs) -> None:
        self.resize_method = cfg.resize_method
        in_features = cfg.in_features

        self.down = nnx.List()
        for mult in cfg.feature_multiples:
            out_features = in_features * mult
            layer = nnx.Dict(
                {
                    "block": TResidualBlock(
                        in_features=in_features,
                        out_features=out_features,
                        time_features=cfg.time_features,
                        act=cfg.act,
                        rngs=rngs,
                    ),
                    "downsample": lambda x: nnx.avg_pool(
                        x, window_shape=(2, 2), strides=(2, 2)
                    ),
                }
            )
            self.down.append(layer)

            in_features = out_features

        self.mid = TResidualBlock(
            in_features=in_features,
            out_features=in_features * 2,
            time_features=cfg.time_features,
            act=cfg.act,
            rngs=rngs,
        )

        in_features *= 2
        feature_multiples = [2] + cfg.feature_multiples[::-1]

        self.up = nnx.List()
        for mult in feature_multiples[:-1]:
            out_features = in_features // mult
            skip_in = out_features
            layer = nnx.Dict(
                {
                    "upsample": nnx.Sequential(
                        nnx.Conv(
                            in_features=in_features,
                            out_features=out_features,
                            kernel_size=(3, 3),
                            padding="same",
                            rngs=rngs,
                        ),
                        nnx.BatchNorm(out_features, rngs=rngs),
                        ACTS[cfg.act],
                    ),
                    "block": TResidualBlock(
                        in_features=out_features + skip_in,
                        out_features=out_features,
                        time_features=cfg.time_features,
                        act=cfg.act,
                        rngs=rngs,
                    ),
                }
            )
            self.up.append(layer)

            in_features = out_features

        self.out = nnx.Sequential(
            nnx.Conv(
                in_features=in_features,
                out_features=cfg.out_features,
                kernel_size=(3, 3),
                padding="same",
                rngs=rngs,
            ),
            nnx.BatchNorm(cfg.out_features, rngs=rngs),
        )

    def __call__(self, x: jax.Array, t: jax.Array) -> jax.Array:
        skip_conns = []
        for layer in self.down:
            x = layer["block"](x, t)
            skip_conns.append(x.copy())
            x = layer["downsample"](x)

        x = self.mid(x, t)

        for skip, layer in zip(skip_conns, self.up):
            x = jax.image.resize(
                x,
                shape=(x.shape[0], x.shape[1] * 2, x.shape[2] * 2, x.shape[3]),
                method=self.resize_method,
            )
            x = jnp.concatenate([skip, layer["upsample"](x)], axis=-1)

            x = layer["block"](x, t)

        return self.out(x)
