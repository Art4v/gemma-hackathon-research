"""Differential Attention with causal masking, learned lambda reparameterization, and head-wise RMSNorm."""

import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'diff_attention_causal_group_rmsnorm',
    'batch': 4,
    'heads': 16,
    'seq_len': 4096,
    'head_dim': 64,
    'lambda_init': 0.8,
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k_q, k_k, k_v, k_lq1, k_lk1, k_lq2, k_lk2 = jax.random.split(key, 7)
    b = CONFIG['batch']
    h = CONFIG['heads']
    s = CONFIG['seq_len']
    d = CONFIG['head_dim']
    q = jax.random.normal(k_q, (b, h, s, 2 * d), dtype=dtype)
    k = jax.random.normal(k_k, (b, h, s, 2 * d), dtype=dtype)
    v = jax.random.normal(k_v, (b, h, s, 2 * d), dtype=dtype)
    lambda_q1 = jax.random.normal(k_lq1, (h, d), dtype=dtype) * 0.02
    lambda_k1 = jax.random.normal(k_lk1, (h, d), dtype=dtype) * 0.02
    lambda_q2 = jax.random.normal(k_lq2, (h, d), dtype=dtype) * 0.02
    lambda_k2 = jax.random.normal(k_lk2, (h, d), dtype=dtype) * 0.02
    rms_weight = jnp.ones((h, 2 * d), dtype=dtype)
    return q, k, v, lambda_q1, lambda_k1, lambda_q2, lambda_k2, rms_weight


def workload(q, k, v, lambda_q1, lambda_k1, lambda_q2, lambda_k2, rms_weight):
    d = CONFIG['head_dim']
    s = CONFIG['seq_len']
    lambda_init = CONFIG['lambda_init']
    eps = CONFIG['eps']
    scale = 1.0 / jnp.sqrt(jnp.float32(d))

    q1, q2 = jnp.split(q, 2, axis=-1)
    k1, k2 = jnp.split(k, 2, axis=-1)

    logits1 = jnp.einsum('bhsd,bhtd->bhst', q1.astype(jnp.float32), k1.astype(jnp.float32)) * scale
    logits2 = jnp.einsum('bhsd,bhtd->bhst', q2.astype(jnp.float32), k2.astype(jnp.float32)) * scale

    idx = jnp.arange(s)
    causal_mask = idx[:, None] >= idx[None, :]
    neg_inf = jnp.finfo(jnp.float32).min
    logits1 = jnp.where(causal_mask[None, None, :, :], logits1, neg_inf)
    logits2 = jnp.where(causal_mask[None, None, :, :], logits2, neg_inf)

    a1 = jax.nn.softmax(logits1, axis=-1)
    a2 = jax.nn.softmax(logits2, axis=-1)

    lam1 = jnp.exp(jnp.sum(lambda_q1.astype(jnp.float32) * lambda_k1.astype(jnp.float32), axis=-1))
    lam2 = jnp.exp(jnp.sum(lambda_q2.astype(jnp.float32) * lambda_k2.astype(jnp.float32), axis=-1))
    lam = (lam1 - lam2 + lambda_init)[None, :, None, None]

    diff_attn = a1 - lam * a2
    out = jnp.einsum('bhst,bhtd->bhsd', diff_attn, v.astype(jnp.float32))

    var = jnp.mean(jnp.square(out), axis=-1, keepdims=True)
    out_norm = out * jax.lax.rsqrt(var + eps)
    out_norm = out_norm * rms_weight.astype(jnp.float32)[None, :, None, :]
    out_final = out_norm * (1.0 - lambda_init)
    return out_final.astype(q.dtype)
