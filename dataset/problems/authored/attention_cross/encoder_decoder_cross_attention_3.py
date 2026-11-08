"""Encoder-decoder cross attention with per-example key padding mask."""

import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'encoder_decoder_cross_attention',
    'batch_size': 16,
    'q_seq_len': 1024,
    'kv_seq_len': 2048,
    'num_heads': 32,
    'head_dim': 128,
    'min_kv_len': 512,
    'mask_value': -1e9,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k_q, k_k, k_v, k_len = jax.random.split(key, 4)
    B = CONFIG['batch_size']
    Tq = CONFIG['q_seq_len']
    Tkv = CONFIG['kv_seq_len']
    H = CONFIG['num_heads']
    D = CONFIG['head_dim']

    q = jax.random.normal(k_q, (B, Tq, H, D), dtype=dtype)
    k = jax.random.normal(k_k, (B, Tkv, H, D), dtype=dtype)
    v = jax.random.normal(k_v, (B, Tkv, H, D), dtype=dtype)
    kv_lengths = jax.random.randint(
        k_len, (B,), CONFIG['min_kv_len'], Tkv + 1, dtype=jnp.int32
    )
    return q, k, v, kv_lengths


def workload(q, k, v, kv_lengths):
    Tq = q.shape[1]
    Tkv = k.shape[1]
    D = q.shape[-1]
    scale = D ** -0.5

    q_f32 = q.astype(jnp.float32) * scale
    k_f32 = k.astype(jnp.float32)
    v_f32 = v.astype(jnp.float32)

    # Compute attention logits: (B, H, Tq, Tkv)
    logits = jnp.einsum('bthd,bshd->bhts', q_f32, k_f32)

    # Build per-example key padding mask: (B, 1, 1, Tkv)
    kv_positions = jnp.arange(Tkv, dtype=jnp.int32)[None, None, None, :]
    valid_mask = kv_positions < kv_lengths[:, None, None, None]
    logits = jnp.where(valid_mask, logits, CONFIG['mask_value'])

    # Softmax in float32
    max_logits = jnp.max(logits, axis=-1, keepdims=True)
    exp_logits = jnp.exp(logits - max_logits)
    probs = exp_logits / jnp.sum(exp_logits, axis=-1, keepdims=True)

    # Weighted sum over values: (B, Tq, H, D)
    out = jnp.einsum('bhts,bshd->bthd', probs, v_f32)
    return out.astype(q.dtype)
