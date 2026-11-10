"""[bf16 variant] DNNModel_12341: (linear -> ReLU) x 3 -> linear applied on the last axis of a 4D tensor."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'DNNModel_12341',
    'batch': 8,
    'height': 128,
    'width': 128,
    'in_features': 512,
    'hidden_features': 512,
    'out_features': 384,
}


def create_inputs(dtype=jnp.bfloat16):
    keys = jax.random.split(jax.random.key(0), 9)
    n_in = CONFIG['in_features']
    n_hidden = CONFIG['hidden_features']
    n_out = CONFIG['out_features']
    x = jax.random.uniform(
        keys[0],
        (CONFIG['batch'], CONFIG['height'], CONFIG['width'], n_in),
        dtype,
    )
    w1 = jax.random.normal(keys[1], (n_in, n_hidden), dtype) * 0.02
    b1 = jax.random.normal(keys[2], (n_hidden,), dtype) * 0.02
    w2 = jax.random.normal(keys[3], (n_hidden, n_hidden), dtype) * 0.02
    b2 = jax.random.normal(keys[4], (n_hidden,), dtype) * 0.02
    w3 = jax.random.normal(keys[5], (n_hidden, n_hidden), dtype) * 0.02
    b3 = jax.random.normal(keys[6], (n_hidden,), dtype) * 0.02
    w4 = jax.random.normal(keys[7], (n_hidden, n_out), dtype) * 0.02
    b4 = jax.random.normal(keys[8], (n_out,), dtype) * 0.02
    return x, w1, b1, w2, b2, w3, b3, w4, b4


def workload(x, w1, b1, w2, b2, w3, b3, w4, b4):
    h = jax.nn.relu(x @ w1 + b1)
    h = jax.nn.relu(h @ w2 + b2)
    h = jax.nn.relu(h @ w3 + b3)
    return h @ w4 + b4
