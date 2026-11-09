"""DDM_Encoder: Linear -> ELU -> Linear -> ELU -> Linear -> tanh over the last axis."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'DDM_Encoder_8523',
    'batch': 8,
    'd1': 128,
    'd2': 128,
    'obs_space': 512,
    'dim': 512,
    'hidden': 32 * 3 * 3,
}


def create_inputs(dtype=jnp.float32):
    k = jax.random.split(jax.random.key(0), 4)
    obs, dim, hid = CONFIG['obs_space'], CONFIG['dim'], CONFIG['hidden']
    x = jax.random.uniform(k[0], (CONFIG['batch'], CONFIG['d1'], CONFIG['d2'], obs), dtype)
    w1 = jax.random.normal(k[1], (obs, dim), dtype) * 0.02
    b1 = jnp.zeros((dim,), dtype)
    w2 = jax.random.normal(k[2], (dim, hid), dtype) * 0.02
    b2 = jnp.zeros((hid,), dtype)
    w3 = jax.random.normal(k[3], (hid, dim), dtype) * 0.02
    b3 = jnp.zeros((dim,), dtype)
    return x, w1, b1, w2, b2, w3, b3


def workload(x, w1, b1, w2, b2, w3, b3):
    h = jax.nn.elu(x @ w1 + b1)
    h = jax.nn.elu(h @ w2 + b2)
    return jnp.tanh(h @ w3 + b3)
