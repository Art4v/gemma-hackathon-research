"""[bf16 variant] SimpleNN: Linear -> ReLU -> Linear -> ReLU -> Linear over the last axis."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'SimpleNN_13175',
    'x_shape': (32, 64, 128, 128),  # last axis is input_dim
    'input_dim': 128,
    'hidden1': 50,
    'hidden2': 100,
    'out_dim': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    keys = jax.random.split(jax.random.key(0), 7)
    d_in, h1, h2, d_out = CONFIG['input_dim'], CONFIG['hidden1'], CONFIG['hidden2'], CONFIG['out_dim']
    x = jax.random.uniform(keys[0], CONFIG['x_shape'], dtype)
    w1 = jax.random.normal(keys[1], (d_in, h1), dtype) * 0.02
    b1 = jax.random.normal(keys[2], (h1,), dtype) * 0.02
    w2 = jax.random.normal(keys[3], (h1, h2), dtype) * 0.02
    b2 = jax.random.normal(keys[4], (h2,), dtype) * 0.02
    w3 = jax.random.normal(keys[5], (h2, d_out), dtype) * 0.02
    b3 = jax.random.normal(keys[6], (d_out,), dtype) * 0.02
    return (x, w1, b1, w2, b2, w3, b3)


def workload(x, w1, b1, w2, b2, w3, b3):
    h = jnp.maximum(x @ w1 + b1, 0.0)
    h = jnp.maximum(h @ w2 + b2, 0.0)
    return h @ w3 + b3
