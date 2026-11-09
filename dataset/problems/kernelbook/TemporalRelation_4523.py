"""[bf16 variant] TemporalRelation: Linear(no bias) over features, then (shifted-prev + shifted-next along time)/2 + residual."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'TemporalRelation_4523',
    'time_steps': 32,
    'dim1': 16,
    'dim2': 256,
    'feat_dim': 512,
    'time_window': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    k_x, k_w = jax.random.split(jax.random.key(0))
    feats = jax.random.uniform(
        k_x,
        (CONFIG['time_steps'], CONFIG['dim1'], CONFIG['dim2'], CONFIG['feat_dim']),
        dtype,
    )
    w = jax.random.normal(k_w, (CONFIG['feat_dim'], CONFIG['feat_dim']), dtype) * 0.02
    return feats, w


def workload(feats, w):
    att = feats @ w
    t_len = att.shape[0]
    t = jnp.arange(t_len)
    prev_idx = jnp.where(t < CONFIG['time_window'], 0, t - 1)
    next_idx = jnp.minimum(t + 1, t_len - 1)
    prev = jnp.take(att, prev_idx, axis=0)
    nxt = jnp.take(att, next_idx, axis=0)
    return (prev + nxt) / 2 + feats
