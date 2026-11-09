"""[bf16 variant] SirenLayer: Linear (last dim) -> sin(w0 * x)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'SirenLayer_7825',
    'batch': 8,
    'h': 128,
    'w': 128,
    'in_f': 512,
    'out_f': 512,
    'w0': 30.0,
    'is_last': False,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    kx, kw, kb = jax.random.split(key, 3)
    x = jax.random.uniform(
        kx, (CONFIG['batch'], CONFIG['h'], CONFIG['w'], CONFIG['in_f']), dtype)
    w = jax.random.normal(kw, (CONFIG['in_f'], CONFIG['out_f']), dtype) * 0.02
    b = jax.random.normal(kb, (CONFIG['out_f'],), dtype) * 0.02
    return x, w, b


def workload(x, w, b):
    y = jnp.matmul(x, w) + b
    if CONFIG['is_last']:
        return y
    return jnp.sin(CONFIG['w0'] * y)
