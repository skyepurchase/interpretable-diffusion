import jax.numpy as jnp


def accuracy(
    model,
    images_test,
    label_test
):
    pred = model(images_test).argmax(axis=1)
    correct = jnp.count_nonzero(pred == label_test)
    total = len(label_test)
    accuracy = correct / total

    print(
        f"{correct} labels correctly predicted out of {total}:"
            f" accuracy = {accuracy:.2%}"
    )
