"""makeStyle: global avg_pool2d -> flatten -> L2 normalize over channels."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'makeStyle_6403',
    'batch': 8,
    'channels': 512,
    'height': 128,
    'width': 128,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x0 = jax.random.uniform(
        key,
        (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']),
        dtype=dtype,
    )
    return (x0,)


def workload(x0):
    # Full-window avg_pool2d + flatten == mean over H, W -> (N, C).
    style = jnp.mean(x0, axis=(-2, -1))
    # No eps, matching the PyTorch module.
    return style / jnp.sqrt(jnp.sum(style ** 2, axis=1, keepdims=True))
