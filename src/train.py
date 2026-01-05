import jax
import optax
from flax import nnx


def loss_fun(
    model: nnx.Module,
    data: jax.Array,
    labels: jax.Array
):
    logits = model(data)
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=labels
    ).mean()

    return loss, logits


@nnx.jit
def train_step(
    model: nnx.Module,
    optimizer: nnx.ModelAndOptimizer,
    data: jax.Array,
    labels: jax.Array
):
    loss_gradient = nnx.grad(loss_fun, has_aux=True)
    grads, _ = loss_gradient(model, data, labels)
    optimizer.update(grads)


def train(
    model: nnx.Module,
    optimizer: nnx.ModelAndOptimizer,
    train: jax.Array,
    train_labels: jax.Array,
    val: jax.Array,
    val_labels: jax.Array,
    epochs: int = 300
):
    for i in range(epochs+1):
        train_step(model, optimizer, train, train_labels)
        if i % 50 == 0:
            loss, _ = loss_fun(model, val, val_labels)
            print(f"Epoch {i}: loss={loss:.2f}")
