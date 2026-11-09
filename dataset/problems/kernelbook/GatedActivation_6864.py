"""GatedActivation: chunk(2, dim=1) -> tanh(x) * sigmoid(y)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'GatedActivation_6864',
    'batch': 16,
    'channels': 256,
    'height': 128,
    'width': 128,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']),
        dtype=dtype,
    )
    return (x,)


def workload(x):
    a, b = jnp.split(x, 2, axis=1)
    return jnp.tanh(a) * jax.nn.sigmoid(b)
