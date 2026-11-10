"""AlphaMish activation: x * tanh((1 + softplus(alpha)) * softplus(x)) on NCHW input with per-channel alpha."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'AlphaMish_3991',
    'batch': 8,
    'in_features': 128,
    'height': 256,
    'width': 256,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        (CONFIG['batch'], CONFIG['in_features'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    alpha = jnp.zeros((CONFIG['in_features'], 1, 1), dtype)
    return x, alpha


def workload(x, alpha):
    return x * jnp.tanh((1 + jax.nn.softplus(alpha)) * jax.nn.softplus(x))
