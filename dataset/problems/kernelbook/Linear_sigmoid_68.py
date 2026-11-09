"""Linear (x @ W + b) over the last dim followed by sigmoid."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Linear_sigmoid_68',
    'd0': 16,
    'd1': 16,
    'd2': 128,
    'dim_in': 1024,
    'dim_out': 1024,
}


def create_inputs(dtype=jnp.float32):
    kx, kw, kb = jax.random.split(jax.random.key(0), 3)
    c = CONFIG
    x = jax.random.uniform(kx, (c['d0'], c['d1'], c['d2'], c['dim_in']), dtype)
    w = jax.random.normal(kw, (c['dim_in'], c['dim_out']), dtype) * 0.02
    b = jax.random.normal(kb, (c['dim_out'],), dtype) * 0.02
    return x, w, b


def workload(x, w, b):
    return jax.nn.sigmoid(x @ w + b)
