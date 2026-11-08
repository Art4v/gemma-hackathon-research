"""Gemma-2 style sliding-window GQA with logit tanh soft-capping and attention sink."""

import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'gemma2_softcap_sliding_window_sink_attention',
    'batch_size': 4,
    'seq_len': 4096,
    'num_q_heads': 32,
    'num_kv_heads': 16,
    'head_dim': 128,
    'window_size': 512,
    'softcap': 50.0,
    'sm_scale': 0.08838834764831845,
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
    sink_logit = jax.random.normal(k_s, (Hq,), dtype=jnp.float32) * 0.02
    return q, k, v, sink_logit


def workload(q, k, v, sink_logit):
    B, T, Hq, D = q.shape
    Hkv = k.shape[2]
    G = Hq // Hkv
    W = CONFIG['window_size']
    softcap = CONFIG['softcap']
    sm_scale = CONFIG['sm_scale']

    q_g = jnp.reshape(q, (B, T, Hkv, G, D))
    q_f32 = q_g.astype(jnp.float32)
    k_f32 = k.astype(jnp.float32)
    v_f32 = v.astype(jnp.float32)

    logits = jnp.einsum('bthgd,bshd->bhgts', q_f32, k_f32) * sm_scale
    logits = softcap * jnp.tanh(logits / softcap)

    q_idx = jnp.arange(T)[:, None]
    k_idx = jnp.arange(T)[None, :]
    mask = (q_idx >= k_idx) & ((q_idx - k_idx) < W)

    logits = jnp.where(mask[None, None, None, :, :], logits, -1e30)

    sink = jnp.reshape(sink_logit.astype(jnp.float32), (1, Hkv, G, 1, 1))
    sink_expanded = jnp.broadcast_to(sink, (B, Hkv, G, T, 1))
    logits_with_sink = jnp.concatenate([logits, sink_expanded], axis=-1)

    max_logits = jnp.max(logits_with_sink, axis=-1, keepdims=True)
    exp_logits = jnp.exp(logits_with_sink - max_logits)
    denom = jnp.sum(exp_logits, axis=-1, keepdims=True)
    probs = exp_logits[..., :T] / denom

    out = jnp.einsum('bhgts,bshd->bthgd', probs, v_f32)
    out = jnp.reshape(out, (B, T, Hq, D))
    return out.astype(q.dtype)
