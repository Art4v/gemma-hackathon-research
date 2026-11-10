"""[bf16 variant] LSTM cell step: concat, four affine maps, sigmoid/tanh gates, cell and hidden state update."""

import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'LSTM_11841',
    'batch': 2048,
    'input_size': 2048,
    'cell_size': 2048,
    'hidden_size': 2048,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.bfloat16):
    batch = CONFIG['batch']
    input_size = CONFIG['input_size']
    hidden_size = CONFIG['hidden_size']
    scale = CONFIG['weight_scale']
    in_features = input_size + hidden_size
    keys = jax.random.split(jax.random.key(0), 11)

    x = jax.random.uniform(keys[0], (batch, input_size), dtype)
    hidden = jax.random.uniform(keys[1], (batch, hidden_size), dtype)
    cell = jax.random.uniform(keys[2], (batch, hidden_size), dtype)

    w_f = jax.random.normal(keys[3], (in_features, hidden_size), dtype) * scale
    b_f = jax.random.normal(keys[4], (hidden_size,), dtype) * scale
    w_i = jax.random.normal(keys[5], (in_features, hidden_size), dtype) * scale
    b_i = jax.random.normal(keys[6], (hidden_size,), dtype) * scale
    w_o = jax.random.normal(keys[7], (in_features, hidden_size), dtype) * scale
    b_o = jax.random.normal(keys[8], (hidden_size,), dtype) * scale
    w_c = jax.random.normal(keys[9], (in_features, hidden_size), dtype) * scale
    b_c = jax.random.normal(keys[10], (hidden_size,), dtype) * scale

    return (x, hidden, cell, w_f, b_f, w_i, b_i, w_o, b_o, w_c, b_c)


def workload(x, hidden, cell, w_f, b_f, w_i, b_i, w_o, b_o, w_c, b_c):
    combined = jnp.concatenate((x, hidden), axis=1)
    f = jax.nn.sigmoid(combined @ w_f + b_f)
    i = jax.nn.sigmoid(combined @ w_i + b_i)
    o = jax.nn.sigmoid(combined @ w_o + b_o)
    c = jnp.tanh(combined @ w_c + b_c)
    new_cell = f * cell + i * c
    new_hidden = o * jnp.tanh(new_cell)
    return new_hidden, new_cell
