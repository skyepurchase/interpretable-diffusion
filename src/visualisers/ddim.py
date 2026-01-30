import importlib.util

import jax
import jax.numpy as jnp
from diffuse.denoisers.denoiser import Denoiser
from diffuse.diffusion.sde import Flow
from diffuse.integrator.deterministic import DDIMIntegrator
from diffuse.predictor import Predictor
from diffuse.timer.base import VpTimer
from flax import nnx, serialization
from huggingface_hub import hf_hub_download

REPO_ID = "jcopo/mnist"


def generate(*args, key: jax.Array) -> jax.Array:
    model_path = hf_hub_download(repo_id=REPO_ID, filename="model.msgpack")
    config_path = hf_hub_download(repo_id=REPO_ID, filename="config.py")

    spec = importlib.util.spec_from_file_location("model_config", config_path)
    assert spec is not None
    assert spec.loader is not None

    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)

    model = config_module.model

    with open(model_path, "rb") as f:
        state_dict = serialization.from_bytes(None, f.read())

    graphdef, state = nnx.split(model)
    state.replace_by_pure_dict(state_dict)
    model = nnx.merge(graphdef, state)
    model.eval()

    flow = Flow()

    predictor = Predictor(
        model=flow, network=lambda x, t: model(x, t).output, prediction_type="velocity"
    )

    img_shape = (28, 28, 1)
    n_steps = 50
    n_samples = 100

    timer = VpTimer(eps=1e-3, tf=1.0, n_steps=n_steps)

    ddim = DDIMIntegrator(model=flow, timer=timer)

    ddim_denoiser = Denoiser(
        integrator=ddim,
        model=flow,
        predictor=predictor,
        x0_shape=img_shape,
    )

    key, key_ddim = jax.random.split(key, 2)
    state, _ = ddim_denoiser.generate(key_ddim, n_steps, n_samples, keep_history=True)

    return jnp.clip(state.integrator_state.position, 0, 1)
