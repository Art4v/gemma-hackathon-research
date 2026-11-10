"""Actor MLP: Linear -> tanh -> Linear -> ReLU -> Linear -> tanh."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Actor_1583',
    'batch': 8,
    'dim1': 128,
    'dim2': 128,
    's_dim': 256,
    'hidden1': 512,
    'hidden2': 384,
    'a_dim': 256,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.float32):
    keys = jax.random.split(jax.random.key(0), 7)
    scale = CONFIG['weight_scale']
    s_dim = CONFIG['s_dim']
    hidden1 = CONFIG['hidden1']
    hidden2 = CONFIG['hidden2']
    a_dim = CONFIG['a_dim']
    x = jax.random.uniform(
        keys[0],
        (CONFIG['batch'], CONFIG['dim1'], CONFIG['dim2'], s_dim),
        dtype,
    )
    w1 = jax.random.normal(keys[1], (s_dim, hidden1), dtype) * scale
    b1 = jax.random.normal(keys[2], (hidden1,), dtype) * scale
    w2 = jax.random.normal(keys[3], (hidden1, hidden2), dtype) * scale
    b2 = jax.random.normal(keys[4], (hidden2,), dtype) * scale
    w3 = jax.random.normal(keys[5], (hidden2, a_dim), dtype) * scale
    b3 = jax.random.normal(keys[6], (a_dim,), dtype) * scale
    return x, w1, b1, w2, b2, w3, b3


def workload(x, w1, b1, w2, b2, w3, b3):
    h = jnp.tanh(x @ w1 + b1)
    h = jax.nn.relu(h @ w2 + b2)
    return jnp.tanh(h @ w3 + b3)
