"""MediatorNet: Linear -> ReLU -> Linear -> ReLU -> Linear (over the last dim)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'MediatorNet_5379',
    'batch': 8,
    'dim1': 32,
    'dim2': 128,
    'input_dim': 1024,
    'hidden_mult': 3,
    'out_dim': 2,
}


def create_inputs(dtype=jnp.float32):
    d = CONFIG['input_dim']
    h = d * CONFIG['hidden_mult']
    o = CONFIG['out_dim']
    keys = jax.random.split(jax.random.key(0), 7)
    x = jax.random.uniform(keys[0], (CONFIG['batch'], CONFIG['dim1'], CONFIG['dim2'], d), dtype)
    w1 = jax.random.normal(keys[1], (d, h), dtype) * 0.02
    b1 = jax.random.normal(keys[2], (h,), dtype) * 0.02
    w2 = jax.random.normal(keys[3], (h, h), dtype) * 0.02
    b2 = jax.random.normal(keys[4], (h,), dtype) * 0.02
    w3 = jax.random.normal(keys[5], (h, o), dtype) * 0.02
    b3 = jax.random.normal(keys[6], (o,), dtype) * 0.02
    return x, w1, b1, w2, b2, w3, b3


def workload(x, w1, b1, w2, b2, w3, b3):
    h = jax.nn.relu(x @ w1 + b1)
    h = jax.nn.relu(h @ w2 + b2)
    return h @ w3 + b3
