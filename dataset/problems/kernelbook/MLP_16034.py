"""MLP: reshape -> Linear+ReLU -> Linear+ReLU -> Linear+ReLU."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'MLP_16034',
    'batch': 16384,
    'in_features': 32 * 32 * 3,
    'hidden': 512,
    'num_class': 10,
}


def create_inputs(dtype=jnp.float32):
    keys = jax.random.split(jax.random.key(0), 7)
    b, d, h, c = CONFIG['batch'], CONFIG['in_features'], CONFIG['hidden'], CONFIG['num_class']
    x = jax.random.uniform(keys[0], (b, d), dtype)
    w1 = jax.random.normal(keys[1], (d, h), dtype) * 0.02
    b1 = jax.random.normal(keys[2], (h,), dtype) * 0.02
    w2 = jax.random.normal(keys[3], (h, h), dtype) * 0.02
    b2 = jax.random.normal(keys[4], (h,), dtype) * 0.02
    w3 = jax.random.normal(keys[5], (h, c), dtype) * 0.02
    b3 = jax.random.normal(keys[6], (c,), dtype) * 0.02
    return x, w1, b1, w2, b2, w3, b3


def workload(x, w1, b1, w2, b2, w3, b3):
    x = x.reshape(-1, CONFIG['in_features'])
    x = jax.nn.relu(x @ w1 + b1)
    x = jax.nn.relu(x @ w2 + b2)
    return jax.nn.relu(x @ w3 + b3)
