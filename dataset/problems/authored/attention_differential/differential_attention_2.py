"""Causal differential attention: two softmax maps from split Q/K half-heads, (A1 - lambda*A2) V, per-head RMSNorm, scale by (1 - lambda_init)."""
import math

import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'differential_attention',
    'batch': 2,
    'num_heads': 16,        # differential heads; Q/K have 2 * num_heads half-heads
    'seq_len': 4096,
    'head_dim': 64,         # half-head dim for Q/K; V head dim is 2 * head_dim
    'layer_idx': 12,        # used for lambda_init = 0.8 - 0.6 * exp(-0.3 * (layer_idx - 1))
    'lambda_init_base': 0.8,
    'lambda_init_amp': 0.6,
    'lambda_init_decay': 0.3,
    'rms_eps': 1e-5,
    'causal': True,
}


def create_inputs(dtype=jnp.bfloat16):
    B = CONFIG['batch']
    H = CONFIG['num_heads']
    S = CONFIG['seq_len']
    d = CONFIG['head_dim']
    key = jax.random.key(0)
    kq, kk, kv, k1, k2, k3, k4 = jax.random.split(key, 7)
    q = jax.random.normal(kq, (B, 2 * H, S, d), dtype)
    k = jax.random.normal(kk, (B, 2 * H, S, d), dtype)
    v = jax.random.normal(kv, (B, H, S, 2 * d), dtype)
    lambda_q1 = jax.random.normal(k1, (d,), dtype) * 0.02
    lambda_k1 = jax.random.normal(k2, (d,), dtype) * 0.02
    lambda_q2 = jax.random.normal(k3, (d,), dtype) * 0.02
    lambda_k2 = jax.random.normal(k4, (d,), dtype) * 0.02
    subln_weight = jnp.ones((2 * d,), dtype)
    return q, k, v, lambda_q1, lambda_k1, lambda_q2, lambda_k2, subln_weight


def workload(q, k, v, lambda_q1, lambda_k1, lambda_q2, lambda_k2, subln_weight):
    B, H2, S, d = q.shape
    H = H2 // 2
    dv = v.shape[-1]
    out_dtype = q.dtype
    scale = 1.0 / math.sqrt(d)

    lambda_init = CONFIG['lambda_init_base'] - CONFIG['lambda_init_amp'] * math.exp(
        -CONFIG['lambda_init_decay'] * (CONFIG['layer_idx'] - 1))
    lam1 = jnp.exp(jnp.sum(lambda_q1.astype(jnp.float32) * lambda_k1.astype(jnp.float32)))
    lam2 = jnp.exp(jnp.sum(lambda_q2.astype(jnp.float32) * lambda_k2.astype(jnp.float32)))
    lam = lam1 - lam2 + lambda_init

    # Half-heads (2h, 2h+1) form differential head h.
    qh = q.reshape(B, H, 2, S, d).reshape(B * H, 2, S, d)
    kh = k.reshape(B, H, 2, S, d).reshape(B * H, 2, S, d)
    vh = v.reshape(B * H, S, dv)

    if CONFIG['causal']:
        mask = jnp.tril(jnp.ones((S, S), dtype=bool))
    else:
        mask = jnp.ones((S, S), dtype=bool)
    w = subln_weight.astype(jnp.float32)
    eps = CONFIG['rms_eps']

    def one_head(args):
        qi, ki, vi = args  # [2,S,d], [2,S,d], [S,dv]
        s = jnp.einsum('tqd,tkd->tqk', qi, ki,
                       preferred_element_type=jnp.float32) * scale
        s = jnp.where(mask[None], s, -jnp.inf)
        p = jax.nn.softmax(s, axis=-1)
        a = p[0] - lam * p[1]
        o = jnp.einsum('qk,kd->qd', a, vi.astype(jnp.float32),
                       preferred_element_type=jnp.float32)
        ms = jnp.mean(o * o, axis=-1, keepdims=True)
        o = o * lax.rsqrt(ms + eps) * w
        o = o * (1.0 - lambda_init)
        return o.astype(out_dtype)

    out = lax.map(one_head, (qh, kh, vh))
    return out.reshape(B, H, S, dv)
