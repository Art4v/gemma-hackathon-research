"""AgentA2C: (Linear -> ReLU) x 2 followed by separate Linear logits and Linear value heads."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'AgentA2C_7363',
    'batch': 8,
    'dim1': 128,
    'dim2': 128,
    'state_dim': 128,
    'hidden': 256,
    'n_actions': 128,
    'value_dim': 1,
    'param_scale': 0.02,
}


def create_inputs(dtype=jnp.float32):
    keys = jax.random.split(jax.random.key(0), 9)
    scale = CONFIG['param_scale']
    state_dim = CONFIG['state_dim']
    hidden = CONFIG['hidden']
    n_actions = CONFIG['n_actions']
    value_dim = CONFIG['value_dim']
    state = jax.random.uniform(
        keys[0],
        (CONFIG['batch'], CONFIG['dim1'], CONFIG['dim2'], state_dim),
        dtype,
    )
    w1 = jax.random.normal(keys[1], (state_dim, hidden), dtype) * scale
    b1 = jax.random.normal(keys[2], (hidden,), dtype) * scale
    w2 = jax.random.normal(keys[3], (hidden, hidden), dtype) * scale
    b2 = jax.random.normal(keys[4], (hidden,), dtype) * scale
    w_logits = jax.random.normal(keys[5], (hidden, n_actions), dtype) * scale
    b_logits = jax.random.normal(keys[6], (n_actions,), dtype) * scale
    w_value = jax.random.normal(keys[7], (hidden, value_dim), dtype) * scale
    b_value = jax.random.normal(keys[8], (value_dim,), dtype) * scale
    return state, w1, b1, w2, b2, w_logits, b_logits, w_value, b_value


def workload(state, w1, b1, w2, b2, w_logits, b_logits, w_value, b_value):
    h = jax.nn.relu(state @ w1 + b1)
    h = jax.nn.relu(h @ w2 + b2)
    logits = h @ w_logits + b_logits
    value = h @ w_value + b_value
    return logits, value
