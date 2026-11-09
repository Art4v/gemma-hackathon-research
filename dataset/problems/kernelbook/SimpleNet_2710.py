"""SimpleNet: (Linear -> tanh) x4 -> Linear over the last axis."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'SimpleNet_2710',
    'batch': 16,
    'd1': 64,
    'd2': 128,
    'ni': 128,
    'hidden': [128, 128, 64, 64, 1],
}


def create_inputs(dtype=jnp.float32):
    dims = [CONFIG['ni']] + CONFIG['hidden']
    keys = jax.random.split(jax.random.key(0), 1 + 2 * (len(dims) - 1))
    x = jax.random.uniform(
        keys[0], (CONFIG['batch'], CONFIG['d1'], CONFIG['d2'], CONFIG['ni']), dtype)
    params = []
    for i in range(len(dims) - 1):
        w = jax.random.normal(keys[1 + 2 * i], (dims[i], dims[i + 1]), dtype) * 0.02
        b = jax.random.normal(keys[2 + 2 * i], (dims[i + 1],), dtype) * 0.02
        params.extend([w, b])
    return (x, *params)


def workload(x, w1, b1, w2, b2, w3, b3, w4, b4, w5, b5):
    x = jnp.tanh(x @ w1 + b1)
    x = jnp.tanh(x @ w2 + b2)
    x = jnp.tanh(x @ w3 + b3)
    x = jnp.tanh(x @ w4 + b4)
    return x @ w5 + b5
