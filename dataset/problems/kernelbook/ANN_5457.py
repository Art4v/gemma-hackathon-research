"""Linear -> ReLU -> Linear -> tanh (ANN decoder MLP on the last axis)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'ANN_5457',
    'lead_shape': (8, 8, 512),
    'input_dim': 1024,
    'hidden_dim': 1024,
    'output_dim': 1024,
    'use_tanh': True,
}


def create_inputs(dtype=jnp.float32):
    k = jax.random.split(jax.random.key(0), 5)
    d_in, d_h, d_out = CONFIG['input_dim'], CONFIG['hidden_dim'], CONFIG['output_dim']
    x = jax.random.uniform(k[0], (*CONFIG['lead_shape'], d_in), dtype)
    w1 = jax.random.normal(k[1], (d_in, d_h), dtype) * 0.02
    b1 = jax.random.normal(k[2], (d_h,), dtype) * 0.02
    w2 = jax.random.normal(k[3], (d_h, d_out), dtype) * 0.02
    b2 = jax.random.normal(k[4], (d_out,), dtype) * 0.02
    return x, w1, b1, w2, b2


def workload(x, w1, b1, w2, b2):
    h = jax.nn.relu(x @ w1 + b1)
    y = h @ w2 + b2
    if CONFIG['use_tanh']:
        y = jnp.tanh(y)
    return y
