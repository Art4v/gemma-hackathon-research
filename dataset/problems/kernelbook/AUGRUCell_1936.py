"""[bf16 variant] AUGRUCell_1936: linear x2 -> chunk(3) x2 -> sigmoid reset/update gates -> tanh new state -> attention-scaled update blend."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'AUGRUCell_1936',
    'batch': 2048,
    'input_size': 2048,
    'hidden_size': 2048,
    'num_gates': 3,
    'att_rows': 512,
    'att_cols': 4,
}


def create_inputs(dtype=jnp.bfloat16):
    keys = jax.random.split(jax.random.key(0), 5)
    batch = CONFIG['batch']
    input_size = CONFIG['input_size']
    hidden_size = CONFIG['hidden_size']
    gate_size = CONFIG['num_gates'] * hidden_size

    x = jax.random.uniform(keys[0], (batch, input_size), dtype)
    hidden = jax.random.uniform(keys[1], (batch, hidden_size), dtype)
    att_score = jax.random.uniform(keys[2], (CONFIG['att_rows'], CONFIG['att_cols']), dtype)

    weight_ih = jax.random.normal(keys[3], (input_size, gate_size), dtype) * 0.02
    weight_hh = jax.random.normal(keys[4], (hidden_size, gate_size), dtype) * 0.02
    bias_ih = jnp.zeros((gate_size,), dtype)
    bias_hh = jnp.zeros((gate_size,), dtype)

    return x, hidden, att_score, weight_ih, weight_hh, bias_ih, bias_hh


def workload(x, hidden, att_score, weight_ih, weight_hh, bias_ih, bias_hh):
    gi = x @ weight_ih + bias_ih
    gh = hidden @ weight_hh + bias_hh

    i_r, i_u, i_h = jnp.split(gi, CONFIG['num_gates'], axis=1)
    h_r, h_u, h_h = jnp.split(gh, CONFIG['num_gates'], axis=1)

    reset_gate = jax.nn.sigmoid(i_r + h_r)
    update_gate = jax.nn.sigmoid(i_u + h_u)
    new_state = jnp.tanh(i_h + reset_gate * h_h)

    update_gate = att_score.reshape(-1, 1) * update_gate
    return (1 - update_gate) * hidden + update_gate * new_state
