"""PolicyNetwork forward: Linear -> ReLU -> Linear -> ReLU -> (mean Linear, log_std Linear -> clamp)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'PolicyNetwork_4213',
    'batch': 16,
    'seq': 4096,
    'state_dim': 512,
    'hidden_dim': 1024,
    'action_dim': 128,
    'log_std_min': -20.0,
    'log_std_max': 2.0,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    ks = jax.random.split(jax.random.key(0), 9)
    s, h, a = c['state_dim'], c['hidden_dim'], c['action_dim']
    state = jax.random.uniform(ks[0], (c['batch'], c['seq'], s), dtype)
    w1 = jax.random.normal(ks[1], (s, h), dtype) * 0.02
    b1 = jax.random.normal(ks[2], (h,), dtype) * 0.02
    w2 = jax.random.normal(ks[3], (h, h), dtype) * 0.02
    b2 = jax.random.normal(ks[4], (h,), dtype) * 0.02
    wm = jax.random.normal(ks[5], (h, a), dtype) * 0.02
    bm = jax.random.normal(ks[6], (a,), dtype) * 0.02
    ws = jax.random.normal(ks[7], (h, a), dtype) * 0.02
    bs = jax.random.normal(ks[8], (a,), dtype) * 0.02
    return state, w1, b1, w2, b2, wm, bm, ws, bs


def workload(state, w1, b1, w2, b2, wm, bm, ws, bs):
    x = jax.nn.relu(state @ w1 + b1)
    x = jax.nn.relu(x @ w2 + b2)
    mean = x @ wm + bm
    log_std = jnp.clip(x @ ws + bs, CONFIG['log_std_min'], CONFIG['log_std_max'])
    return mean, log_std
