"""[bf16 variant] Linear -> ReLU -> Linear -> ReLU -> Linear MLP over the last axis."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Net_3568',
    'x_shape': (16, 64, 256, 128),
    'observations_dim': 128,
    'hidden_dim': 512,
    'actions_dim': 128,
}


def create_inputs(dtype=jnp.bfloat16):
    keys = jax.random.split(jax.random.key(0), 7)
    obs = CONFIG['observations_dim']
    hid = CONFIG['hidden_dim']
    act = CONFIG['actions_dim']
    x = jax.random.uniform(keys[0], CONFIG['x_shape'], dtype)
    w0 = jax.random.normal(keys[1], (obs, hid), dtype) * 0.02
    b0 = jax.random.normal(keys[2], (hid,), dtype) * 0.02
    w1 = jax.random.normal(keys[3], (hid, hid), dtype) * 0.02
    b1 = jax.random.normal(keys[4], (hid,), dtype) * 0.02
    w2 = jax.random.normal(keys[5], (hid, act), dtype) * 0.02
    b2 = jax.random.normal(keys[6], (act,), dtype) * 0.02
    return x, w0, b0, w1, b1, w2, b2


def workload(x, w0, b0, w1, b1, w2, b2):
    h = jax.nn.relu(x @ w0 + b0)
    h = jax.nn.relu(h @ w1 + b1)
    return h @ w2 + b2
