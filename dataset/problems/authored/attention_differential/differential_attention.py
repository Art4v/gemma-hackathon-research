"""Differential Attention with dual causal softmax and headwise RMSNorm."""

import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'differential_attention',
    'batch_size': 4,
    'num_heads': 16,
    'seq_len': 4096,
    'head_dim': 64,
    'lambda_init': 0.8,
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k_q1, k_q2, k_k1, k_k2, k_v, k_lq1, k_lk1, k_lq2, k_lk2 = jax.random.split(key, 9)
    B = CONFIG['batch_size']
    H = CONFIG['num_heads']
    S = CONFIG['seq_len']
    D = CONFIG['head_dim']

    q1 = jax.random.normal(k_q1, (B, H, S, D), dtype=dtype)
    q2 = jax.random.normal(k_q2, (B, H, S, D), dtype=dtype)
    k1 = jax.random.normal(k_k1, (B, H, S, D), dtype=dtype)
    k2 = jax.random.normal(k_k2, (B, H, S, D), dtype=dtype)
    v = jax.random.normal(k_v, (B, H, S, 2 * D), dtype=dtype)

    lambda_q1 = jax.random.normal(k_lq1, (H, D), dtype=dtype) * 0.02
    lambda_k1 = jax.random.normal(k_lk1, (H, D), dtype=dtype) * 0.02
    lambda_q2 = jax.random.normal(k_lq2, (H, D), dtype=dtype) * 0.02
    lambda_k2 = jax.random.normal(k_lk2, (H, D), dtype=dtype) * 0.02
    rms_weight = jnp.ones((H, 2 * D), dtype=dtype)

    return q1, q2, k1, k2, v, lambda_q1, lambda_k1, lambda_q2, lambda_k2, rms_weight


def workload(q1, q2, k1, k2, v, lambda_q1, lambda_k1, lambda_q2, lambda_k2, rms_weight):
    S = CONFIG['seq_len']
    D = CONFIG['head_dim']
    lambda_init = CONFIG['lambda_init']
    eps = CONFIG['eps']
    scale = 1.0 / jnp.sqrt(jnp.float32(D))

    q1_f32 = q1.astype(jnp.float32)
    q2_f32 = q2.astype(jnp.float32)
    k1_f32 = k1.astype(jnp.float32)
    k2_f32 = k2.astype(jnp.float32)
    v_f32 = v.astype(jnp.float32)

    logits1 = jnp.matmul(q1_f32, jnp.swapaxes(k1_f32, -1, -2)) * scale
    logits2 = jnp.matmul(q2_f32, jnp.swapaxes(k2_f32, -1, -2)) * scale

    mask = jnp.tril(jnp.ones((S, S), dtype=jnp.bool_))
    neg_inf = jnp.finfo(jnp.float32).min
    logits1 = jnp.where(mask, logits1, neg_inf)
    logits2 = jnp.where(mask, logits2, neg_inf)

    a1 = jax.nn.softmax(logits1, axis=-1)
    a2 = jax.nn.softmax(logits2, axis=-1)

    lq1 = lambda_q1.astype(jnp.float32)
    lk1 = lambda_k1.astype(jnp.float32)
    lq2 = lambda_q2.astype(jnp.float32)
    lk2 = lambda_k2.astype(jnp.float32)

    lambda_1 = jnp.exp(jnp.sum(lq1 * lk1, axis=-1))
    lambda_2 = jnp.exp(jnp.sum(lq2 * lk2, axis=-1))
    lambda_full = (lambda_1 - lambda_2 + lambda_init)[:, None, None]

    diff_attn = a1 - lambda_full * a2
    out = jnp.matmul(diff_attn, v_f32)

    variance = jnp.mean(jnp.square(out), axis=-1, keepdims=True)
    out_norm = out * jax.lax.rsqrt(variance + eps)
    out_norm = out_norm * rms_weight.astype(jnp.float32)[:, None, :]
    out_norm = out_norm * (1.0 - lambda_init)

    return out_norm.astype(q1.dtype)
