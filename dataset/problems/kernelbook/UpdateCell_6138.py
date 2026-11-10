"""[bf16 variant] UpdateCell: affine(x) + linear(hidden), chunk in two, sigmoid/tanh gates, gated blend with hidden."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'UpdateCell_6138',
    'batch': 8192,
    'input_dim': 1024,
    'output_dim': 1024,
    'num_chunks': 2,
    'chunk_axis': 1,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.bfloat16):
    batch = CONFIG['batch']
    input_dim = CONFIG['input_dim']
    output_dim = CONFIG['output_dim']
    projected_dim = CONFIG['num_chunks'] * output_dim
    scale = CONFIG['weight_scale']

    key = jax.random.key(0)
    k_x, k_hidden, k_w_x2i, k_b_x2i, k_w_h2h = jax.random.split(key, 5)

    x = jax.random.uniform(k_x, (batch, input_dim), dtype)
    hidden = jax.random.uniform(k_hidden, (batch, output_dim), dtype)
    w_x2i = jax.random.normal(k_w_x2i, (input_dim, projected_dim), dtype) * scale
    b_x2i = jax.random.normal(k_b_x2i, (projected_dim,), dtype) * scale
    w_h2h = jax.random.normal(k_w_h2h, (output_dim, projected_dim), dtype) * scale
    return x, hidden, w_x2i, b_x2i, w_h2h


def workload(x, hidden, w_x2i, b_x2i, w_h2h):
    num_chunks = CONFIG['num_chunks']
    axis = CONFIG['chunk_axis']

    i_i, i_n = jnp.split(x @ w_x2i + b_x2i, num_chunks, axis=axis)
    h_i, h_n = jnp.split(hidden @ w_h2h, num_chunks, axis=axis)
    input_gate = jax.nn.sigmoid(i_i + h_i)
    new_gate = jnp.tanh(i_n + h_n)
    return new_gate + input_gate * (hidden - new_gate)
