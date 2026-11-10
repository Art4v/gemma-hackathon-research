"""CoreNetwork_10707: two affine projections (i2h, h2h) on the last axis, summed, then ReLU."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'CoreNetwork_10707',
    'batch': 8,
    'dim_1': 128,
    'dim_2': 128,
    'input_size': 256,
    'hidden_size': 256,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.float32):
    keys = jax.random.split(jax.random.key(0), 6)
    leading = (CONFIG['batch'], CONFIG['dim_1'], CONFIG['dim_2'])
    input_size = CONFIG['input_size']
    hidden_size = CONFIG['hidden_size']
    scale = CONFIG['weight_scale']

    g_t = jax.random.uniform(keys[0], leading + (input_size,), dtype)
    h_t_prev = jax.random.uniform(keys[1], leading + (hidden_size,), dtype)
    w_i2h = jax.random.normal(keys[2], (input_size, hidden_size), dtype) * scale
    b_i2h = jax.random.normal(keys[3], (hidden_size,), dtype) * scale
    w_h2h = jax.random.normal(keys[4], (hidden_size, hidden_size), dtype) * scale
    b_h2h = jax.random.normal(keys[5], (hidden_size,), dtype) * scale
    return g_t, h_t_prev, w_i2h, b_i2h, w_h2h, b_h2h


def workload(g_t, h_t_prev, w_i2h, b_i2h, w_h2h, b_h2h):
    h1 = g_t @ w_i2h + b_i2h
    h2 = h_t_prev @ w_h2h + b_h2h
    return jax.nn.relu(h1 + h2)
