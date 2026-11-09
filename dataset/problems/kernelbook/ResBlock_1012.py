"""ResBlock: Linear -> ReLU -> Linear -> residual add."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'ResBlock_1012',
    'batch': 8,
    'dim1': 64,
    'dim2': 64,
    'size': 1024,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    kx, k1, kb1, k2, kb2 = jax.random.split(key, 5)
    size = CONFIG['size']
    x = jax.random.uniform(kx, (CONFIG['batch'], CONFIG['dim1'], CONFIG['dim2'], size), dtype)
    w1 = jax.random.normal(k1, (size, size), dtype) * 0.02
    b1 = jax.random.normal(kb1, (size,), dtype) * 0.02
    w2 = jax.random.normal(k2, (size, size), dtype) * 0.02
    b2 = jax.random.normal(kb2, (size,), dtype) * 0.02
    return x, w1, b1, w2, b2


def workload(x, w1, b1, w2, b2):
    h = jax.nn.relu(x @ w1 + b1)
    return h @ w2 + b2 + x
