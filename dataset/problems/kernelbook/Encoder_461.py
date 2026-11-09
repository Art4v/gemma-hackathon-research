"""Encoder: tanh(Linear) -> two Linear heads (mean, log_var)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Encoder_461',
    'batch': 8,
    'dim1': 32,
    'dim2': 256,
    'input_size': 512,
    'hidden_size': 1024,
    'bottleneck': 256,
}


def create_inputs(dtype=jnp.float32):
    keys = jax.random.split(jax.random.key(0), 7)
    c = CONFIG
    x = jax.random.uniform(keys[0], (c['batch'], c['dim1'], c['dim2'], c['input_size']), dtype)
    w1 = jax.random.normal(keys[1], (c['input_size'], c['hidden_size']), dtype) * 0.02
    b1 = jax.random.normal(keys[2], (c['hidden_size'],), dtype) * 0.02
    wm = jax.random.normal(keys[3], (c['hidden_size'], c['bottleneck']), dtype) * 0.02
    bm = jax.random.normal(keys[4], (c['bottleneck'],), dtype) * 0.02
    wv = jax.random.normal(keys[5], (c['hidden_size'], c['bottleneck']), dtype) * 0.02
    bv = jax.random.normal(keys[6], (c['bottleneck'],), dtype) * 0.02
    return x, w1, b1, wm, bm, wv, bv


def workload(x, w1, b1, wm, bm, wv, bv):
    # PyTorch runs linear1 twice; both calls return the same values, so compute it once.
    h = jnp.tanh(x @ w1 + b1)
    mean = h @ wm + bm
    log_var = h @ wv + bv
    return mean, log_var
