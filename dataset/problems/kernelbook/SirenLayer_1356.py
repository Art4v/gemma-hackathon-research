"""SirenLayer: Linear (last dim) -> sin(w0 * x)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'SirenLayer_1356',
    'batch': 8,
    'd1': 64,
    'd2': 128,
    'in_f': 512,
    'out_f': 512,
    'w0': 30.0,
    'is_last': False,
}


def create_inputs(dtype=jnp.float32):
    k_x, k_w, k_b = jax.random.split(jax.random.key(0), 3)
    x = jax.random.uniform(
        k_x, (CONFIG['batch'], CONFIG['d1'], CONFIG['d2'], CONFIG['in_f']), dtype)
    w = jax.random.normal(k_w, (CONFIG['in_f'], CONFIG['out_f']), dtype) * 0.02
    b = jax.random.normal(k_b, (CONFIG['out_f'],), dtype) * 0.02
    return x, w, b


def workload(x, w, b):
    y = jnp.matmul(x, w) + b
    if CONFIG['is_last']:
        return y
    return jnp.sin(CONFIG['w0'] * y)
