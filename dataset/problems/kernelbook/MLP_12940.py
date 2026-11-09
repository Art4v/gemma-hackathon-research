"""MLP_12940: Linear -> tanh -> Linear -> tanh -> Linear -> tanh over the last axis."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'MLP_12940',
    'lead_shape': (8, 128, 128),
    'n_inputs': 128,
    'hidden': 40,
    'n_outputs': 4,
}


def create_inputs(dtype=jnp.float32):
    keys = jax.random.split(jax.random.key(0), 7)
    n_in, hid, n_out = CONFIG['n_inputs'], CONFIG['hidden'], CONFIG['n_outputs']
    x = jax.random.uniform(keys[0], (*CONFIG['lead_shape'], n_in), dtype)
    w1 = jax.random.normal(keys[1], (n_in, hid), dtype) * 0.02
    b1 = jax.random.normal(keys[2], (hid,), dtype) * 0.02
    w2 = jax.random.normal(keys[3], (hid, hid), dtype) * 0.02
    b2 = jax.random.normal(keys[4], (hid,), dtype) * 0.02
    w3 = jax.random.normal(keys[5], (hid, n_out), dtype) * 0.02
    b3 = jax.random.normal(keys[6], (n_out,), dtype) * 0.02
    return x, w1, b1, w2, b2, w3, b3


def workload(x, w1, b1, w2, b2, w3, b3):
    h = jnp.tanh(x @ w1 + b1)
    h = jnp.tanh(h @ w2 + b2)
    return jnp.tanh(h @ w3 + b3)
