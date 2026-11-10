"""FC_1124: linear projection on the last axis (x @ w, no bias) followed by LeakyReLU(0.2)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'FC_1124',
    'batch': 8,
    'height': 128,
    'width': 128,
    'in_features': 256,
    'out_features': 256,
    'negative_slope': 0.2,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    k_x, k_w = jax.random.split(key)
    x = jax.random.uniform(
        k_x,
        (CONFIG['batch'], CONFIG['height'], CONFIG['width'], CONFIG['in_features']),
        dtype,
    )
    w = jax.random.normal(
        k_w, (CONFIG['in_features'], CONFIG['out_features']), dtype
    ) * CONFIG['weight_scale']
    return x, w


def workload(x, w):
    y = x @ w
    return jnp.where(y > 0, y, CONFIG['negative_slope'] * y)
