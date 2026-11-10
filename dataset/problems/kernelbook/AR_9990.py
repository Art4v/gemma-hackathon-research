"""AR_9990: transpose(1, 2) -> Linear(window, 1) on last axis -> transpose(1, 2)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'AR_9990',
    'batch': 16,
    'channels': 128,
    'steps': 128,
    'window': 128,
    'out_features': 1,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    k_x, k_w, k_b = jax.random.split(key, 3)
    x = jax.random.uniform(
        k_x,
        (CONFIG['batch'], CONFIG['channels'], CONFIG['steps'], CONFIG['window']),
        dtype,
    )
    w = jax.random.normal(k_w, (CONFIG['window'], CONFIG['out_features']), dtype) * CONFIG['weight_scale']
    b = jax.random.normal(k_b, (CONFIG['out_features'],), dtype) * CONFIG['weight_scale']
    return x, w, b


def workload(x, w, b):
    y = jnp.swapaxes(x, 1, 2)
    y = y @ w + b
    return jnp.swapaxes(y, 1, 2)
