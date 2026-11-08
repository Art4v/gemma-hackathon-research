"""Encoder-decoder cross attention over padded encoder memory with a per-example key padding mask."""

import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'encoder_decoder_cross_attention',
    'batch': 16,
    'q_len': 512,
    'kv_len': 1536,
    'num_heads': 32,
    'head_dim': 128,
    'min_kv_len': 256,
    'mask_value': -1e30,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k_q, k_k, k_v, k_len = jax.random.split(key, 4)
    b = CONFIG['batch']
    q_len = CONFIG['q_len']
    kv_len = CONFIG['kv_len']
    h = CONFIG['num_heads']
    d = CONFIG['head_dim']

    q = jax.random.normal(k_q, (b, q_len, h, d), dtype=dtype)
    k = jax.random.normal(k_k, (b, kv_len, h, d), dtype=dtype)
    v = jax.random.normal(k_v, (b, kv_len, h, d), dtype=dtype)
    kv_valid_lens = jax.random.randint(
        k_len, (b,), CONFIG['min_kv_len'], kv_len + 1, dtype=jnp.int32
    )
    return q, k, v, kv_valid_lens


def workload(q, k, v, kv_valid_lens):
    d = CONFIG['head_dim']
    kv_len = CONFIG['kv_len']
    scale = d ** -0.5

    q_f32 = q.astype(jnp.float32) * scale
    k_f32 = k.astype(jnp.float32)
    v_f32 = v.astype(jnp.float32)

    logits = jnp.einsum('bqhd,bkhd->bhqk', q_f32, k_f32)

    kv_positions = jnp.arange(kv_len, dtype=jnp.int32)[None, None, None, :]
    valid_mask = kv_positions < kv_valid_lens[:, None, None, None]
    logits = jnp.where(valid_mask, logits, CONFIG['mask_value'])

    weights = jax.nn.softmax(logits, axis=-1)
    out = jnp.einsum('bhqk,bkhd->bqhd', weights, v_f32)
    return out.astype(q.dtype)
