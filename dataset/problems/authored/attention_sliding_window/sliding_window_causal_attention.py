"""Sliding-window causal multi-head attention with float32 softmax accumulation."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'sliding_window_causal_attention',
    'batch_size': 2,
    'num_heads': 16,
    'seq_len': 8192,
    'head_dim': 128,
    'window_size': 512,
    'mask_value': -1e30,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k_q, k_k, k_v = jax.random.split(key, 3)
    shape = (
        CONFIG['batch_size'],
        CONFIG['num_heads'],
        CONFIG['seq_len'],
        CONFIG['head_dim'],
    )
    q = jax.random.normal(k_q, shape, dtype=dtype)
    k = jax.random.normal(k_k, shape, dtype=dtype)
    v = jax.random.normal(k_v, shape, dtype=dtype)
    return (q, k, v)


def workload(q, k, v):
    seq_len = CONFIG['seq_len']
    head_dim = CONFIG['head_dim']
    window_size = CONFIG['window_size']
    mask_value = CONFIG['mask_value']

    scale = head_dim ** -0.5
    logits = jnp.einsum('bhqd,bhkd->bhqk', q.astype(jnp.float32), k.astype(jnp.float32)) * scale

    q_idx = jnp.arange(seq_len)[:, None]
    k_idx = jnp.arange(seq_len)[None, :]
    causal_mask = q_idx >= k_idx
    window_mask = (q_idx - k_idx) < window_size
    valid_mask = causal_mask & window_mask

    logits = jnp.where(valid_mask[None, None, :, :], logits, mask_value)
    weights = jax.nn.softmax(logits, axis=-1)
    out = jnp.einsum('bhqk,bhkd->bhqd', weights, v.astype(jnp.float32))
    return out.astype(q.dtype)
