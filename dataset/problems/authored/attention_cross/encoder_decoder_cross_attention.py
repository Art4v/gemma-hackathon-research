"""Encoder-decoder cross-attention with per-example key padding mask and grouped query heads."""

import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'encoder_decoder_cross_attention',
    'batch_size': 16,
    'q_seq_len': 512,
    'kv_seq_len': 2048,
    'num_q_heads': 32,
    'num_kv_heads': 8,
    'head_dim': 128,
    'min_kv_len': 256,
    'mask_value': -1e30,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k_q, k_k, k_v, k_len = jax.random.split(key, 4)
    B = CONFIG['batch_size']
    Tq = CONFIG['q_seq_len']
    Tkv = CONFIG['kv_seq_len']
    Hq = CONFIG['num_q_heads']
    Hkv = CONFIG['num_kv_heads']
    D = CONFIG['head_dim']

    q = jax.random.normal(k_q, (B, Tq, Hq, D), dtype=dtype)
    k = jax.random.normal(k_k, (B, Tkv, Hkv, D), dtype=dtype)
    v = jax.random.normal(k_v, (B, Tkv, Hkv, D), dtype=dtype)
    kv_lengths = jax.random.randint(
        k_len, (B,), CONFIG['min_kv_len'], Tkv + 1, dtype=jnp.int32
    )
    return q, k, v, kv_lengths


def workload(q, k, v, kv_lengths):
    B, Tq, Hq, D = q.shape
    _, Tkv, Hkv, _ = k.shape
    num_groups = Hq // Hkv
    scale = 1.0 / jnp.sqrt(jnp.float32(D))

    q_f32 = q.astype(jnp.float32).reshape(B, Tq, Hkv, num_groups, D)
    k_f32 = k.astype(jnp.float32)
    v_f32 = v.astype(jnp.float32)

    logits = jnp.einsum('bthgd,bshd->bhgts', q_f32, k_f32) * scale

    kv_positions = jnp.arange(Tkv, dtype=jnp.int32)[None, :]
    valid_mask = kv_positions < kv_lengths[:, None]
    valid_mask = valid_mask[:, None, None, None, :]

    masked_logits = jnp.where(valid_mask, logits, CONFIG['mask_value'])
    probs = jax.nn.softmax(masked_logits, axis=-1)

    out = jnp.einsum('bhgts,bshd->bthgd', probs, v_f32)
    out = out.reshape(B, Tq, Hq, D)
    return out.astype(q.dtype)
