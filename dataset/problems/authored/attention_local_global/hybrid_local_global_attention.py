"""Hybrid multi-head attention with alternating sliding-window local and full global causal heads with logit soft-capping."""

import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'hybrid_local_global_attention',
    'batch_size': 8,
    'seq_len': 4096,
    'num_heads': 32,
    'head_dim': 128,
    'window_size': 512,
    'softcap': 50.0,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k1, k2, k3 = jax.random.split(key, 3)
    B = CONFIG['batch_size']
    S = CONFIG['seq_len']
    H = CONFIG['num_heads']
    D = CONFIG['head_dim']

    q = jax.random.normal(k1, (B, H, S, D), dtype=dtype)
    k = jax.random.normal(k2, (B, H, S, D), dtype=dtype)
    v = jax.random.normal(k3, (B, H, S, D), dtype=dtype)
    return (q, k, v)


def workload(q, k, v):
    B, H, S, D = q.shape
    window_size = CONFIG['window_size']
    softcap = CONFIG['softcap']
    scale = D ** -0.5

    q_f32 = q.astype(jnp.float32) * scale
    k_f32 = k.astype(jnp.float32)
    v_f32 = v.astype(jnp.float32)

    logits = jnp.matmul(q_f32, jnp.swapaxes(k_f32, -1, -2))
    logits = softcap * jnp.tanh(logits / softcap)

    q_idx = jnp.arange(S)[:, None]
    k_idx = jnp.arange(S)[None, :]

    causal_mask = q_idx >= k_idx
    local_mask = causal_mask & ((q_idx - k_idx) < window_size)

    head_is_local = (jnp.arange(H) % 2 == 0)[:, None, None]
    combined_mask = jnp.where(head_is_local, local_mask[None, :, :], causal_mask[None, :, :])

    logits = jnp.where(combined_mask[None, :, :, :], logits, -1e30)

    max_logits = jnp.max(logits, axis=-1, keepdims=True)
    exp_logits = jnp.exp(logits - max_logits)
    probs = exp_logits / jnp.sum(exp_logits, axis=-1, keepdims=True)

    out = jnp.matmul(probs, v_f32)
    return out.astype(q.dtype)
