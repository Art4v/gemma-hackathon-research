"""SimpleNet: Linear(12288,84) -> ReLU -> Linear(84,50) -> ReLU -> Linear(50,2)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'SimpleNet_3014',
    'batch': 2048,
    'in_features': 12288,
    'hidden1': 84,
    'hidden2': 50,
    'out_features': 2,
}


def create_inputs(dtype=jnp.float32):
    keys = jax.random.split(jax.random.key(0), 7)
    b, d0 = CONFIG['batch'], CONFIG['in_features']
    d1, d2, d3 = CONFIG['hidden1'], CONFIG['hidden2'], CONFIG['out_features']
    x = jax.random.uniform(keys[0], (b, d0), dtype)
    w1 = jax.random.normal(keys[1], (d0, d1), dtype) * 0.02
    b1 = jax.random.normal(keys[2], (d1,), dtype) * 0.02
    w2 = jax.random.normal(keys[3], (d1, d2), dtype) * 0.02
    b2 = jax.random.normal(keys[4], (d2,), dtype) * 0.02
    w3 = jax.random.normal(keys[5], (d2, d3), dtype) * 0.02
    b3 = jax.random.normal(keys[6], (d3,), dtype) * 0.02
    return x, w1, b1, w2, b2, w3, b3


def workload(x, w1, b1, w2, b2, w3, b3):
    x = x.reshape(-1, w1.shape[0])
    x = jax.nn.relu(x @ w1 + b1)
    x = jax.nn.relu(x @ w2 + b2)
    return x @ w3 + b3
