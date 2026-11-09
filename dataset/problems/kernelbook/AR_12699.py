"""AR_12699: transpose(1,2) -> Linear(window, 1) over last dim -> transpose(1,2)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'AR_12699',
    'batch': 8,
    'dim1': 256,
    'dim2': 128,
    'window': 256,
    'out_features': 1,
}


def create_inputs(dtype=jnp.float32):
    kx, kw, kb = jax.random.split(jax.random.key(0), 3)
    x = jax.random.uniform(
        kx, (CONFIG['batch'], CONFIG['dim1'], CONFIG['dim2'], CONFIG['window']), dtype)
    w = jax.random.normal(kw, (CONFIG['window'], CONFIG['out_features']), dtype) * 0.02
    b = jax.random.normal(kb, (CONFIG['out_features'],), dtype) * 0.02
    return x, w, b


def workload(x, w, b):
    x = jnp.swapaxes(x, 1, 2)
    x = x @ w + b
    return jnp.swapaxes(x, 1, 2)
