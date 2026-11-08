"""Grouped-query sliding-window causal attention with learned per-head sink logits."""

import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'gqa_sliding_window_sink_attention',
    'batch_size': 4,
    'seq_len': 4096,
    'num_q_heads': 32,
    'num_kv_heads': 8,
    'head_dim': 128,
    'window_size': 512,
    'sm_scale': 0.08838834764831845,
    'mask_value': -1e30,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k_q, k_k, k_v, k_s = jax.random.split(key, 4)
    B = CONFIG['batch_size']
    T = CONFIG['seq_len']
    Hq = CONFIG['num_q_heads']
    Hkv = CONFIG['num_kv_heads']
    D = CONFIG['head_dim']

    q = jax.random.normal(k_q, (B, T, Hq, D), dtype=dtype)
    k = jax.random.normal(k_k, (B, T, Hkv, D), dtype=dtype)
    v = jax.random.normal(k_v, (B, T, Hkv, D), dtype=dtype)
    sink_logits = jax.random.normal(k_s, (Hq,), dtype=jnp.float32) * 0.02
    return q, k, v, sink_logits


def workload(q, k, v, sink_logits):
    B, T, Hq, D = q.shape
    Hkv = CONFIG['num_kv_heads']
    G = Hq // Hkv
    W = CONFIG['window_size']
    scale = CONFIG['sm_scale']
    mask_val = CONFIG['mask_value']

    q_grouped = q.reshape(B, T, Hkv, G, D).transpose(0, 2, 3, 1, 4)
    k_t = k.transpose(0, 2, 1, 3)
    v_t = v.transpose(0, 2, 1, 3)

    logits = jnp.einsum(
        'bhgqd,bhkd->bhgqk',
        q_grouped.astype(jnp.float32),
        k_t.astype(jnp.float32),
    ) * scale

    q_idx = jnp.arange(T)[:, None]
    k_idx = jnp.arange(T)[None, :]
    valid_mask = (q_idx >= k_idx) & ((q_idx - k_idx) < W)
    logits = jnp.where(valid_mask[None, None, None, :, :], logits, mask_val)

    sink = sink_logits.astype(jnp.float32).reshape(1, Hkv, G, 1, 1)
    max_logits = jnp.maximum(jnp.max(logits, axis=-1, keepdims=True), sink)

    exp_logits = jnp.exp(logits - max_logits)
    exp_sink = jnp.exp(sink - max_logits)
    denom = jnp.sum(exp_logits, axis=-1, keepdims=True) + exp_sink
    weights = exp_logits / denom

    out = jnp.einsum('bhgqk,bhkd->bhgqd', weights, v_t.astype(jnp.float32))
    out = out.transpose(0, 3, 1, 2, 4).reshape(B, T, Hq, D)
    return out.astype(q.dtype)
