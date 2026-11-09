"""make_style: global avg_pool2d over HxW -> flatten -> L2 normalize over channels."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'make_style_12017',
    'batch': 16,
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
    # avg_pool2d with kernel = full spatial extent is a mean over H, W; flatten gives (N, C).
    style = jnp.mean(x0, axis=(2, 3))
    return style / jnp.sqrt(jnp.sum(style ** 2, axis=1, keepdims=True))
