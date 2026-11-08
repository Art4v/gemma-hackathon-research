"""Single-token MQA decode with QK-RMSNorm, logit soft-capping, and sliding-window valid-length masking."""

import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'gemma2_softcap_mqa_decode_sliding_window',
    'batch_size': 64,
    'num_q_heads': 32,
    'max_seq_len': 32768,
    'head_dim': 128,
    'sliding_window': 4096,
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
    min_len = CONFIG['min_valid_len']

    q = jax.random.normal(k_q, (B, H_q, D), dtype=dtype)
    k_cache = jax.random.normal(k_k, (B, S, D), dtype=dtype)
    v_cache = jax.random.normal(k_v, (B, S, D), dtype=dtype)
    seq_lens = jax.random.randint(k_len, (B,), minval=min_len, maxval=S + 1, dtype=jnp.int32)

    q_norm_weight = jnp.ones((D,), dtype=dtype)
    k_norm_weight = jnp.ones((D,), dtype=dtype)

    return q, k_cache, v_cache, seq_lens, q_norm_weight, k_norm_weight


def workload(q, k_cache, v_cache, seq_lens, q_norm_weight, k_norm_weight):
    B, S, D = k_cache.shape
    eps = CONFIG['eps']
    softcap = CONFIG['softcap']
    window = CONFIG['sliding_window']

    q_f32 = q.astype(jnp.float32)
    k_f32 = k_cache.astype(jnp.float32)
    v_f32 = v_cache.astype(jnp.float32)
    qw_f32 = q_norm_weight.astype(jnp.float32)
    kw_f32 = k_norm_weight.astype(jnp.float32)

    q_rms = jnp.sqrt(jnp.mean(jnp.square(q_f32), axis=-1, keepdims=True) + eps)
    q_normed = (q_f32 / q_rms) * qw_f32

    k_rms = jnp.sqrt(jnp.mean(jnp.square(k_f32), axis=-1, keepdims=True) + eps)
    k_normed = (k_f32 / k_rms) * kw_f32

    scale = 1.0 / jnp.sqrt(jnp.float32(D))
    logits = jnp.einsum('bhd,bsd->bhs', q_normed * scale, k_normed)

    logits = softcap * jnp.tanh(logits / softcap)

    pos = jnp.arange(S, dtype=jnp.int32)[None, :]
    valid_len = seq_lens[:, None]
    start_pos = jnp.maximum(0, valid_len - window)
    mask = (pos < valid_len) & (pos >= start_pos)

    logits = jnp.where(mask[:, None, :], logits, -1e30)

    max_logits = jnp.max(logits, axis=-1, keepdims=True)
    exp_logits = jnp.exp(logits - max_logits)
    probs = exp_logits / jnp.sum(exp_logits, axis=-1, keepdims=True)

    out = jnp.einsum('bhs,bsd->bhd', probs, v_f32)
    return out.astype(q.dtype)
