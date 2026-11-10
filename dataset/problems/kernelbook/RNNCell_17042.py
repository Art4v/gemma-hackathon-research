"""RNNCell_17042: relu(linear(x, w_ih, bias) + linear(h, w_hh)) over the last axis."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'RNNCell_17042',
    'batch': 8,
    'dim1': 128,
    'dim2': 128,
    'input_size': 256,
    'hidden_size': 256,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.float32):
    k_x, k_h, k_ih, k_hh, k_b = jax.random.split(jax.random.key(0), 5)
    prefix = (CONFIG['batch'], CONFIG['dim1'], CONFIG['dim2'])
    input_size = CONFIG['input_size']
    hidden_size = CONFIG['hidden_size']
    scale = CONFIG['weight_scale']
    x = jax.random.uniform(k_x, prefix + (input_size,), dtype)
    h = jax.random.uniform(k_h, prefix + (hidden_size,), dtype)
    w_ih = jax.random.normal(k_ih, (input_size, hidden_size), dtype) * scale
    w_hh = jax.random.normal(k_hh, (hidden_size, hidden_size), dtype) * scale
    bias = jax.random.normal(k_b, (hidden_size,), dtype) * scale
    return x, h, w_ih, w_hh, bias


def workload(x, h, w_ih, w_hh, bias):
    output = x @ w_ih + bias + h @ w_hh
    return jnp.maximum(output, jnp.zeros((), output.dtype))
