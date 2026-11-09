"""piNetwork: Linear -> tanh -> Linear -> tanh -> Linear -> softmax(dim=-1)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'piNetwork_12730',
    'batch': 8,
    'dim1': 64,
    'dim2': 128,
    'input_size': 512,
    'hidden_size1': 1024,
    'hidden_size2': 1024,
    'action_size': 256,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    keys = jax.random.split(jax.random.key(0), 7)
    x = jax.random.uniform(
        keys[0], (c['batch'], c['dim1'], c['dim2'], c['input_size']), dtype)
    w1 = jax.random.normal(keys[1], (c['input_size'], c['hidden_size1']), dtype) * 0.02
    b1 = jax.random.normal(keys[2], (c['hidden_size1'],), dtype) * 0.02
    w2 = jax.random.normal(keys[3], (c['hidden_size1'], c['hidden_size2']), dtype) * 0.02
    b2 = jax.random.normal(keys[4], (c['hidden_size2'],), dtype) * 0.02
    w3 = jax.random.normal(keys[5], (c['hidden_size2'], c['action_size']), dtype) * 0.02
    b3 = jax.random.normal(keys[6], (c['action_size'],), dtype) * 0.02
    return x, w1, b1, w2, b2, w3, b3


def workload(x, w1, b1, w2, b2, w3, b3):
    h = jnp.tanh(x @ w1 + b1)
    h = jnp.tanh(h @ w2 + b2)
    return jax.nn.softmax(h @ w3 + b3, axis=-1)
