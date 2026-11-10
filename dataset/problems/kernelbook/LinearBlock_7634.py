"""Linear (x @ w + b) on the last axis followed by ReLU."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'LinearBlock_7634',
    'batch': 8,
    'dim1': 128,
    'dim2': 128,
    'in_dim': 256,
    'out_dim': 256,
}


def create_inputs(dtype=jnp.float32):
    k_x, k_w, k_b = jax.random.split(jax.random.key(0), 3)
    x = jax.random.uniform(
        k_x,
        (CONFIG['batch'], CONFIG['dim1'], CONFIG['dim2'], CONFIG['in_dim']),
        dtype,
    )
    w = jax.random.normal(k_w, (CONFIG['in_dim'], CONFIG['out_dim']), dtype) * 0.02
    b = jax.random.normal(k_b, (CONFIG['out_dim'],), dtype) * 0.02
    return x, w, b


def workload(x, w, b):
    out = jnp.matmul(x, w) + b
    return jnp.maximum(out, jnp.zeros((), out.dtype))
