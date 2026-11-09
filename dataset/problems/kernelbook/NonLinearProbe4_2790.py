"""NonLinearProbe4: Linear -> ReLU -> Linear -> ReLU -> Linear over the last axis."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'NonLinearProbe4_2790',
    'batch': 8,
    'dim1': 32,
    'dim2': 256,
    'input_dim': 512,
    'num_hidden': 512,
    'num_classes': 256,
}


def create_inputs(dtype=jnp.float32):
    keys = jax.random.split(jax.random.key(0), 7)
    c = CONFIG
    x = jax.random.uniform(keys[0], (c['batch'], c['dim1'], c['dim2'], c['input_dim']), dtype)
    w1 = jax.random.normal(keys[1], (c['input_dim'], c['num_hidden']), dtype) * 0.02
    b1 = jax.random.normal(keys[2], (c['num_hidden'],), dtype) * 0.02
    w2 = jax.random.normal(keys[3], (c['num_hidden'], c['num_hidden']), dtype) * 0.02
    b2 = jax.random.normal(keys[4], (c['num_hidden'],), dtype) * 0.02
    w3 = jax.random.normal(keys[5], (c['num_hidden'], c['num_classes']), dtype) * 0.02
    b3 = jax.random.normal(keys[6], (c['num_classes'],), dtype) * 0.02
    return x, w1, b1, w2, b2, w3, b3


def workload(x, w1, b1, w2, b2, w3, b3):
    h = jax.nn.relu(x @ w1 + b1)
    h = jax.nn.relu(h @ w2 + b2)
    return h @ w3 + b3
