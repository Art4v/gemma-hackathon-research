"""[bf16 variant] ValueNetwork_9042: Linear -> ReLU -> Linear -> ReLU -> Linear over the last axis."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'ValueNetwork_9042',
    'batch': 8,
    'dim1': 128,
    'dim2': 128,
    'state_dim': 256,
    'hidden_dim': 256,
    'output_dim': 1,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.bfloat16):
    keys = jax.random.split(jax.random.key(0), 7)
    state_dim = CONFIG['state_dim']
    hidden_dim = CONFIG['hidden_dim']
    output_dim = CONFIG['output_dim']
    scale = CONFIG['weight_scale']
    state = jax.random.uniform(
        keys[0],
        (CONFIG['batch'], CONFIG['dim1'], CONFIG['dim2'], state_dim),
        dtype,
    )
    w1 = jax.random.normal(keys[1], (state_dim, hidden_dim), dtype) * scale
    b1 = jax.random.normal(keys[2], (hidden_dim,), dtype) * scale
    w2 = jax.random.normal(keys[3], (hidden_dim, hidden_dim), dtype) * scale
    b2 = jax.random.normal(keys[4], (hidden_dim,), dtype) * scale
    w3 = jax.random.normal(keys[5], (hidden_dim, output_dim), dtype) * scale
    b3 = jax.random.normal(keys[6], (output_dim,), dtype) * scale
    return state, w1, b1, w2, b2, w3, b3


def workload(state, w1, b1, w2, b2, w3, b3):
    x = jax.nn.relu(state @ w1 + b1)
    x = jax.nn.relu(x @ w2 + b2)
    return x @ w3 + b3
