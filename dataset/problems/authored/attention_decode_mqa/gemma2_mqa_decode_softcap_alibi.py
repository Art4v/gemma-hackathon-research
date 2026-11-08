"""Fused MQA single-token decode with QK-RMSNorm, ALiBi position bias, logit tanh soft-capping, and per-sequence valid length masking."""

import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'gemma2_mqa_decode_softcap_alibi',
    'batch_size': 64,
    'num_q_heads': 32,
    'max_seq_len': 32768,
    'head_dim': 128,
    'softcap': 50.0,
    'eps': 1e-6,
    'min_valid_len': 1024,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k_q, k_k, k_v, k_len = jax.random.split(key, 4)
    B = CONFIG['batch_size']
    H_q = CONFIG['num_q_heads']
    S = CONFIG['max_seq_len']
    D = CONFIG['head_dim']

    q = jax.random.normal(k_q, (B, H_q, D), dtype=dtype)
    k_cache = jax.random.normal(k_k, (B, S, D), dtype=dtype)
    v_cache = jax.random.normal(k_v, (B, S, D), dtype=dtype)
    valid_lens = jax.random.randint(
        k_len, (B,), CONFIG['min_valid_len'], S + 1, dtype=jnp.int32
    )

    q_norm_scale = jnp.ones((D,), dtype=dtype)
    k_norm_scale = jnp.ones((D,), dtype=dtype)
    alibi_slopes = jnp.exp2(-8.0 * jnp.arange(1, H_q + 1, dtype=jnp.float32) / H_q)

    return q, k_cache, v_cache, valid_lens, q_norm_scale, k_norm_scale, alibi_slopes


def workload(q, k_cache, v_cache, valid_lens, q_norm_scale, k_norm_scale, alibi_slopes):
    D = CONFIG['head_dim']
    S = CONFIG['max_seq_len']
    softcap = CONFIG['softcap']
    eps = CONFIG['eps']
    sm_scale = D ** -0.5

    q_f32 = q.astype(jnp.float32)
    q_var = jnp.mean(jnp.square(q_f32), axis=-1, keepdims=True)
    q_normed = q_f32 * jax.lax.rsqrt(q_var + eps) * q_norm_scale.astype(jnp.float32)

    k_f32 = k_cache.astype(jnp.float32)
    k_var = jnp.mean(jnp.square(k_f32), axis=-1, keepdims=True)
    k_normed = k_f32 * jax.lax.rsqrt(k_var + eps) * k_norm_scale.astype(jnp.float32)

    logits = jnp.einsum('bhd,bsd->bhs', q_normed, k_normed) * sm_scale
    logits = softcap * jnp.tanh(logits / softcap)

    pos = jnp.arange(S, dtype=jnp.int32)[None, :]
    val_len = valid_lens[:, None]
    rel_pos = (pos - (val_len - 1)).astype(jnp.float32)
    alibi_bias = alibi_slopes[None, :, None] * rel_pos[:, None, :]
    logits = logits + alibi_bias

    mask = pos < val_len
    logits = jnp.where(mask[:, None, :], logits, -1e30)

    max_logits = jnp.max(logits, axis=-1, keepdims=True)
    exp_logits = jnp.where(mask[:, None, :], jnp.exp(logits - max_logits), 0.0)
    probs = exp_logits / jnp.sum(exp_logits, axis=-1, keepdims=True)

    out = jnp.einsum('bhs,bsd->bhd', probs, v_cache.astype(jnp.float32))
    return out.astype(q.dtype)
