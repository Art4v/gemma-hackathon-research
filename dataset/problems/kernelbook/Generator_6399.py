"""Generator MLP: Linear -> ReLU -> Linear -> ReLU -> Linear -> sigmoid over the last dim."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Generator_6399',
    'batch': 8,
    'dim1': 32,
    'dim2': 128,
    'input_size': 1024,
    'hidden_size': 1024,
    'output_size': 1024,
}


def create_inputs(dtype=jnp.float32):
    keys = jax.random.split(jax.random.key(0), 7)
    c = CONFIG
    x = jax.random.uniform(keys[0], (c['batch'], c['dim1'], c['dim2'], c['input_size']), dtype)
    w1 = jax.random.normal(keys[1], (c['input_size'], c['hidden_size']), dtype) * 0.02
    b1 = jax.random.normal(keys[2], (c['hidden_size'],), dtype) * 0.02
    w2 = jax.random.normal(keys[3], (c['hidden_size'], c['hidden_size']), dtype) * 0.02
    b2 = jax.random.normal(keys[4], (c['hidden_size'],), dtype) * 0.02
    w3 = jax.random.normal(keys[5], (c['hidden_size'], c['output_size']), dtype) * 0.02
    b3 = jax.random.normal(keys[6], (c['output_size'],), dtype) * 0.02
    return x, w1, b1, w2, b2, w3, b3


def workload(x, w1, b1, w2, b2, w3, b3):
    h = jax.nn.relu(x @ w1 + b1)
    h = jax.nn.relu(h @ w2 + b2)
    return jax.nn.sigmoid(h @ w3 + b3)
