"""Grouped-query attention with learned per-head attention sink logits and causal sliding window."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'gqa_sliding_window_sink_attention',
    'batch_size': 4,
    'seq_len': 4096,
    'num_q_heads': 32,
    'num_kv_heads': 8,
    'head_dim': 128,
    'window_size': 1024,
    'mask_value': -1e30,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k_q, k_k, k_v, k_s = jax.random.split(key, 4)
    B = CONFIG['batch_size']
    S = CONFIG['seq_len']
    H_Q = CONFIG['num_q_heads']
    H_KV = CONFIG['num_kv_heads']
    D = CONFIG['head_dim']

    q = jax.random.normal(k_q, (B, S, H_Q, D), dtype=dtype)
    k = jax.random.normal(k_k, (B, S, H_KV, D), dtype=dtype)
    v = jax.random.normal(k_v, (B, S, H_KV, D), dtype=dtype)
    sink_logits = jax.random.normal(k_s, (H_Q,), dtype=jnp.float32) * 0.02
    return q, k, v, sink_logits


def workload(q, k, v, sink_logits):
    B, S, H_Q, D = q.shape
    H_KV = k.shape[2]
    G = H_Q // H_KV
    W = CONFIG['window_size']
    scale = D ** -0.5

    q_grouped = q.reshape(B, S, H_KV, G, D).transpose(0, 2, 3, 1, 4).astype(jnp.float32)
    k_t = k.transpose(0, 2, 1, 3).astype(jnp.float32)
    v_t = v.transpose(0, 2, 1, 3).astype(jnp.float32)

    logits = jnp.einsum('bhgqd,bhkd->bhgqk', q_grouped * scale, k_t)

    q_idx = jnp.arange(S)[:, None]
    k_idx = jnp.arange(S)[None, :]
    valid_mask = (q_idx >= k_idx) & ((q_idx - k_idx) < W)
    logits = jnp.where(valid_mask[None, None, None, :, :], logits, CONFIG['mask_value'])

    sinks = sink_logits.astype(jnp.float32).reshape(1, H_KV, G, 1, 1)
    max_logits = jnp.maximum(jnp.max(logits, axis=-1, keepdims=True), sinks)

    exp_logits = jnp.exp(logits - max_logits)
    exp_sinks = jnp.exp(sinks - max_logits)
    denom = jnp.sum(exp_logits, axis=-1, keepdims=True) + exp_sinks

    weights = exp_logits / denom
    out = jnp.einsum('bhgqk,bhkd->bhgqd', weights, v_t)
    out = out.transpose(0, 3, 1, 2, 4).reshape(B, S, H_Q, D)
    return out.astype(q.dtype)
