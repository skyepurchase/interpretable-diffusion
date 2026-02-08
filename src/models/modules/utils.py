from typing import Callable

import jax

ModuleFn = Callable[[jax.Array], jax.Array]
