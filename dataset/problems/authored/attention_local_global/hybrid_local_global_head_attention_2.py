"""Alternating local sliding-window and global causal multi-head attention."""

import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'hybrid_local_global_head_attention',
    'batch_size': 4,
    'seq_len': 4096,
    'num_heads': 32,
    'head_dim': 128,
    'window_size': 256,
    'sm_scale': 0.08838834764831845,
    'neg_inf': -1e30,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k1, k2, k3 = jax.random.split(key, 3)
    b = CONFIG['batch_size']
    s = CONFIG['seq_len']
    h = CONFIG['num_heads']
    d = CONFIG['head_dim']
    q = jax.random.normal(k1, (b, h, s, d), dtype=dtype)
    k = jax.random.normal(k2, (b, h, s, d), dtype=dtype)
    v = jax.random.normal(k3, (b, h, s, d), dtype=dtype)
    return (q, k, v)


def workload(q, k, v):
    s = q.shape[2]
    h = q.shape[1]
    window_size = CONFIG['window_size']
    sm_scale = CONFIG['sm_scale']
    neg_inf = CONFIG['neg_inf']

    logits = jnp.einsum('bhsd,bhtd->bhst', q.astype(jnp.float32), k.astype(jnp.float32)) * sm_scale

    q_idx = jnp.arange(s)[:, None]
    k_idx = jnp.arange(s)[None, :]

    causal_mask = q_idx >= k_idx
    local_mask = causal_mask & ((q_idx - k_idx) < window_size)

    head_idx = jnp.arange(h)[:, None, None]
    is_local_head = (head_idx % 2) == 0

    combined_mask = jnp.where(is_local_head, local_mask[None, :, :], causal_mask[None, :, :])
    masked_logits = jnp.where(combined_mask[None, :, :, :], logits, neg_inf)

    probs = jax.nn.softmax(masked_logits, axis=-1)
    out = jnp.einsum('bhst,bhtd->bhsd', probs, v.astype(jnp.float32))
    return out.astype(q.dtype)
