"""Causal differential attention: two softmax maps from split Q/K sub-heads, (A1 - lambda*A2) V, per-head RMSNorm (SubLN), scaled by (1 - lambda_init)."""
import math

import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'diff_attention_causal_subln',
    'batch': 2,
    'num_heads': 16,          # output heads; Q/K have 2 * num_heads sub-heads
    'seq_len': 4096,
    'head_dim': 64,           # Q/K sub-head dim d; V head dim is 2 * d
    'layer_idx': 12,          # depth used for lambda_init
    'lambda_init_base': 0.8,
    'lambda_init_amp': 0.6,
    'lambda_init_decay': 0.3,
    'rms_eps': 1e-5,
    'q_chunk': 512,
    'param_std': 0.02,
}


def create_inputs(dtype=jnp.bfloat16):
    B = CONFIG['batch']
    H = CONFIG['num_heads']
    S = CONFIG['seq_len']
    D = CONFIG['head_dim']
    key = jax.random.key(0)
    kq, kk, kv, k1, k2, k3, k4 = jax.random.split(key, 7)
    q = jax.random.normal(kq, (B, 2 * H, S, D), dtype)
    k = jax.random.normal(kk, (B, 2 * H, S, D), dtype)
    v = jax.random.normal(kv, (B, H, S, 2 * D), dtype)
    std = CONFIG['param_std']
    lambda_q1 = jax.random.normal(k1, (D,), dtype) * std
    lambda_k1 = jax.random.normal(k2, (D,), dtype) * std
    lambda_q2 = jax.random.normal(k3, (D,), dtype) * std
    lambda_k2 = jax.random.normal(k4, (D,), dtype) * std
    subln_weight = jnp.ones((2 * D,), dtype)
    return q, k, v, lambda_q1, lambda_k1, lambda_q2, lambda_k2, subln_weight


def workload(q, k, v, lambda_q1, lambda_k1, lambda_q2, lambda_k2, subln_weight):
    B, H2, S, D = q.shape
    H = H2 // 2
    Dv = v.shape[-1]
    C = CONFIG['q_chunk']
    n_chunks = S // C
    eps = CONFIG['rms_eps']
    scale = 1.0 / math.sqrt(D)

    lambda_init = CONFIG['lambda_init_base'] - CONFIG['lambda_init_amp'] * math.exp(
        -CONFIG['lambda_init_decay'] * (CONFIG['layer_idx'] - 1))
    f32 = jnp.float32
    lam = (jnp.exp(jnp.sum(lambda_q1.astype(f32) * lambda_k1.astype(f32)))
           - jnp.exp(jnp.sum(lambda_q2.astype(f32) * lambda_k2.astype(f32)))
           + lambda_init)

    v32 = v.astype(f32)
    w32 = subln_weight.astype(f32)
    k_pos = jnp.arange(S)

    # (n_chunks, B, 2H, C, D)
    q_chunks = q.reshape(B, H2, n_chunks, C, D).transpose(2, 0, 1, 3, 4)

    def body(args):
        qb, idx = args
        s = jnp.einsum('bhqd,bhkd->bhqk', qb, k, preferred_element_type=f32) * scale
        q_pos = idx * C + jnp.arange(C)
        mask = q_pos[:, None] >= k_pos[None, :]
        s = jnp.where(mask[None, None], s, -jnp.inf)
        p = jax.nn.softmax(s, axis=-1)                       # (B, 2H, C, S)
        p = p.reshape(B, H, 2, C, S)
        a = p[:, :, 0] - lam * p[:, :, 1]                    # (B, H, C, S)
        o = jnp.einsum('bhqk,bhkd->bhqd', a, v32)            # (B, H, C, Dv)
        ms = jnp.mean(o * o, axis=-1, keepdims=True)
        o = o * lax.rsqrt(ms + eps) * w32
        o = o * (1.0 - lambda_init)
        return o.astype(q.dtype)

    outs = lax.map(body, (q_chunks, jnp.arange(n_chunks)))  # (n_chunks, B, H, C, Dv)
    out = outs.transpose(1, 0, 3, 2, 4).reshape(B, S, H * Dv)
    return out
