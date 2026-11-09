"""ResnetBlockFC: relu -> linear -> relu -> linear, plus identity residual."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'ResnetBlockFC_7838',
    'batch': 8,
    'd1': 32,
    'd2': 256,
    'size_in': 1024,
    'size_h': 1024,
    'size_out': 1024,
}


def create_inputs(dtype=jnp.float32):
    keys = jax.random.split(jax.random.key(0), 5)
    c = CONFIG
    x = jax.random.uniform(keys[0], (c['batch'], c['d1'], c['d2'], c['size_in']), dtype)
    w0 = jax.random.normal(keys[1], (c['size_in'], c['size_h']), dtype) * 0.02
    b0 = jax.random.normal(keys[2], (c['size_h'],), dtype) * 0.02
    w1 = jax.random.normal(keys[3], (c['size_h'], c['size_out']), dtype) * 0.02
    b1 = jax.random.normal(keys[4], (c['size_out'],), dtype) * 0.02
    return x, w0, b0, w1, b1


def workload(x, w0, b0, w1, b1):
    # size_in == size_out, so the shortcut is the identity.
    net = jax.nn.relu(x) @ w0 + b0
    dx = jax.nn.relu(net) @ w1 + b1
    return x + dx
