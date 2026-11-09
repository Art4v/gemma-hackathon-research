"""[bf16 variant] ValueNetwork: Linear -> ReLU -> Linear (hidden -> 1) over the last axis of a 4D state."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'ValueNetwork_17733',
    'batch': 16,
    'dim1': 32,
    'dim2': 128,
    'num_inputs': 1024,
    'hidden_dim': 1024,
    'out_dim': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    k_x, k_w1, k_b1, k_w2, k_b2 = jax.random.split(jax.random.key(0), 5)
    c = CONFIG
    state = jax.random.uniform(
        k_x, (c['batch'], c['dim1'], c['dim2'], c['num_inputs']), dtype)
    w1 = jax.random.normal(k_w1, (c['num_inputs'], c['hidden_dim']), dtype) * 0.02
    b1 = jax.random.normal(k_b1, (c['hidden_dim'],), dtype) * 0.02
    w2 = jax.random.normal(k_w2, (c['hidden_dim'], c['out_dim']), dtype) * 0.02
    b2 = jax.random.normal(k_b2, (c['out_dim'],), dtype) * 0.02
    return state, w1, b1, w2, b2


def workload(state, w1, b1, w2, b2):
    h = jax.nn.relu(state @ w1 + b1)
    return h @ w2 + b2
