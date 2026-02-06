import jax
import jax.numpy as jnp
from flax import nnx
from tqdm import tqdm

from models.ddim import DDIM


def generate(model: nnx.Module, array_like: jax.Array, key: jax.Array) -> jax.Array:
    """
    Given a trained DDIM model and array who's shape is to be produced generate images.

    Parameters
    ----------
    model : DDIM
        The trained DDIM model
    array_like : jax.Array
        Array who's shape is matched
    key : jax.Array
        PRNG key

    Returns
    images : jax.Array
        The generated images with shape like array_like
    """
    if not isinstance(model, DDIM):
        raise TypeError(f"Expected model to be DDIM but received {type(model)}")
    # model_path = hf_hub_download(repo_id=REPO_ID, filename="model.msgpack")
    # config_path = hf_hub_download(repo_id=REPO_ID, filename="config.py")

    # spec = importlib.util.spec_from_file_location("model_config", config_path)
    # assert spec is not None
    # assert spec.loader is not None

    # config_module = importlib.util.module_from_spec(spec)
    # spec.loader.exec_module(config_module)

    # model = config_module.model

    # with open(model_path, "rb") as f:
    #     state_dict = serialization.from_bytes(None, f.read())

    # graphdef, state = nnx.split(model)
    # state.replace_by_pure_dict(state_dict)
    # model = nnx.merge(graphdef, state)
    model.eval()

    # flow = Flow()

    # predictor = Predictor(
    #     model=flow, network=lambda x, t: model(x, t), prediction_type="noise"
    # )

    img_shape = array_like.shape[1:]
    n_samples = array_like.shape[0]
    # n_steps = 50

    # # timer = VpTimer(eps=1e-3, tf=1.0, n_steps=n_steps)
    # timer = DDIMTimer(n_steps=n_steps, n_time_training=model.num_timesteps)

    # ddim = DDIMIntegrator(model=flow, timer=timer)

    # ddim_denoiser = Denoiser(
    #     integrator=ddim,
    #     model=flow,
    #     predictor=predictor,
    #     x0_shape=img_shape,
    # )

    steps = [999] + [10 * i for i in range(90, -1, -1)]
    alphas = (1 - model.betas).cumprod(axis=0)
    print(len(steps))
    pred_x = array_like.copy()
    for step in tqdm(steps, desc="Sampling"):
        key, noise_key = jax.random.split(key, 2)
        random_start = jax.random.normal(noise_key, shape=(n_samples, *img_shape))

        alpha = alphas[step].reshape(-1, 1, 1, 1)
        x = pred_x * jnp.sqrt(alpha) + random_start * jnp.sqrt(1 - alpha)

        eps = model(x, step * jnp.ones(n_samples))
        pred_x = (x - eps * jnp.sqrt(1.0 - alpha)) / jnp.sqrt(alpha)

    # key, state_key = jax.random.split(key, 2)
    # keys = jax.random.split(state_key, n_samples)
    # state = jax.vmap(ddim_denoiser.init, in_axes=(0, 0))(random_start, keys)

    # state = ddim_denoiser.step(state)
    # jax.debug.breakpoint()

    # state, _ = ddim_denoiser.generate(key_ddim, n_steps, n_samples, keep_history=True)

    return jnp.clip(pred_x, 0, 1)
