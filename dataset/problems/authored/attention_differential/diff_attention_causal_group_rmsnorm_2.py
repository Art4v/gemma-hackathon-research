"""Differential Attention with causal masking, reparameterized lambda, and per-head RMSNorm."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'diff_attention_causal_group_rmsnorm',
    'batch': 8,
    'heads': 16,
    'seq_len': 2048,
    'head_dim': 64,
    'v_head_dim': 128,
    'lambda_init': 0.8,
    'eps': 1e-5,
}

def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k_q1, k_q2, k_k1, k_k2, k_v, k_lq1, k_lk1, k_lq2, k_lk2 = jax.random.split(key, 9)
    B = CONFIG['batch']
    H = CONFIG['heads']
    S = CONFIG['seq_len']
    D = CONFIG['head_dim']
    Dv = CONFIG['v_head_dim']

    q1 = jax.random.normal(k_q1, (B, H, S, D), dtype=dtype)
    q2 = jax.random.normal(k_q2, (B, H, S, D), dtype=dtype)
    k1 = jax.random.normal(k_k1, (B, H, S, D), dtype=dtype)
    k2 = jax.random.normal(k_k2, (B, H, S, D), dtype=dtype)
    v = jax.random.normal(k_v, (B, H, S, Dv), dtype=dtype)

    lambda_q1 = jax.random.normal(k_lq1, (H, D), dtype=dtype) * 0.02
    lambda_k1 = jax.random.normal(k_lk1, (H, D), dtype=dtype) * 0.02
    lambda_q2 = jax.random.normal(k_lq2, (H, D), dtype=dtype) * 0.02
    lambda_k2 = jax.random.normal(k_lk2, (H, D), dtype=dtype) * 0.02
    norm_weight = jnp.ones((H, Dv), dtype=dtype)

    return q1, q2, k1, k2, v, lambda_q1, lambda_k1, lambda_q2, lambda_k2, norm_weight

def workload(q1, q2, k1, k2, v, lambda_q1, lambda_k1, lambda_q2, lambda_k2, norm_weight):
    S = CONFIG['seq_len']
    D = CONFIG['head_dim']
    lambda_init = CONFIG['lambda_init']
    eps = CONFIG['eps']
    scale = 1.0 / jnp.sqrt(jnp.float32(D))

    q1_f = q1.astype(jnp.float32)
    q2_f = q2.astype(jnp.float32)
    k1_f = k1.astype(jnp.float32)
    k2_f = k2.astype(jnp.float32)
    v_f = v.astype(jnp.float32)

    logits1 = jnp.einsum('bhsd,bhtd->bhst', q1_f, k1_f) * scale
    logits2 = jnp.einsum('bhsd,bhtd->bhst', q2_f, k2_f) * scale

    idx = jnp.arange(S)
    causal_mask = idx[:, None] >= idx[None, :]
    neg_inf = jnp.finfo(jnp.float32).min
    logits1 = jnp.where(causal_mask[None, None, :, :], logits1, neg_inf)
    logits2 = jnp.where(causal_mask[None, None, :, :], logits2, neg_inf)

    a1 = jax.nn.softmax(logits1, axis=-1)
    a2 = jax.nn.softmax(logits2, axis=-1)

    l1 = jnp.exp(jnp.sum(lambda_q1.astype(jnp.float32) * lambda_k1.astype(jnp.float32), axis=-1))
    l2 = jnp.exp(jnp.sum(lambda_q2.astype(jnp.float32) * lambda_k2.astype(jnp.float32), axis=-1))
    lam = (l1 - l2 + lambda_init)[None, :, None, None]

    diff_attn = a1 - lam * a2
    out = jnp.einsum('bhst,bhtd->bhsd', diff_attn, v_f)

    var = jnp.mean(jnp.square(out), axis=-1, keepdims=True)
    out_norm = out * jax.lax.rsqrt(var + eps)
    out_norm = out_norm * norm_weight.astype(jnp.float32)[None, :, None, :] * (1.0 - lambda_init)

    return out_norm.astype(v.dtype)
