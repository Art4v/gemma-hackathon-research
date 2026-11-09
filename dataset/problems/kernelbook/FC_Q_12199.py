"""[bf16 variant] FC_Q: Linear -> ReLU -> Linear -> ReLU -> Linear (q head only; the i head is computed but unused in PyTorch)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'FC_Q_12199',
    'batch': 8,
    'dim1': 256,
    'dim2': 256,
    'state_dim': 128,
    'hidden': 256,
    'num_actions': 128,
}


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG
    ks = jax.random.split(jax.random.key(0), 7)
    state = jax.random.uniform(ks[0], (c['batch'], c['dim1'], c['dim2'], c['state_dim']), dtype)
    w1 = jax.random.normal(ks[1], (c['state_dim'], c['hidden']), dtype) * 0.02
    b1 = jax.random.normal(ks[2], (c['hidden'],), dtype) * 0.02
    w2 = jax.random.normal(ks[3], (c['hidden'], c['hidden']), dtype) * 0.02
    b2 = jax.random.normal(ks[4], (c['hidden'],), dtype) * 0.02
    w3 = jax.random.normal(ks[5], (c['hidden'], c['num_actions']), dtype) * 0.02
    b3 = jax.random.normal(ks[6], (c['num_actions'],), dtype) * 0.02
    return state, w1, b1, w2, b2, w3, b3


def workload(state, w1, b1, w2, b2, w3, b3):
    q = jax.nn.relu(state @ w1 + b1)
    q = jax.nn.relu(q @ w2 + b2)
    return q @ w3 + b3
