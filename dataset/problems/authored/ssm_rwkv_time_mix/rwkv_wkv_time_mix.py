"""RWKV-4 WKV time-mixing recurrence with max-tracked exponential decay and receptance gating."""

import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    "name": "rwkv_wkv_time_mix",
    "batch": 8,
    "seq_len": 2048,
    "channels": 4096,
    "eps": 1e-12,
    "neg_inf": -1e30,
}


def create_inputs(dtype=jnp.bfloat16):
    B = CONFIG["batch"]
    T = CONFIG["seq_len"]
    C = CONFIG["channels"]
    key = jax.random.key(0)
    k_r, k_k, k_v, k_w, k_u = jax.random.split(key, 5)
    r = jax.random.normal(k_r, (B, T, C), dtype=dtype)
    k = jax.random.uniform(k_k, (B, T, C), dtype=dtype, minval=-2.0, maxval=2.0)
    v = jax.random.normal(k_v, (B, T, C), dtype=dtype)
    w = -jax.nn.softplus(jax.random.normal(k_w, (C,), dtype=jnp.float32))
    u = jax.random.normal(k_u, (C,), dtype=jnp.float32) * 0.02
    return (r, k, v, w, u)


def workload(r, k, v, w, u):
    B, T, C = k.shape
    eps = CONFIG["eps"]
    neg_inf = CONFIG["neg_inf"]

    k_f32 = jnp.transpose(k.astype(jnp.float32), (1, 0, 2))
    v_f32 = jnp.transpose(v.astype(jnp.float32), (1, 0, 2))
    w_f32 = w.astype(jnp.float32)[None, :]
    u_f32 = u.astype(jnp.float32)[None, :]

    init_aa = jnp.zeros((B, C), dtype=jnp.float32)
    init_bb = jnp.zeros((B, C), dtype=jnp.float32)
    init_pp = jnp.full((B, C), neg_inf, dtype=jnp.float32)

    def step(carry, inputs):
        aa, bb, pp = carry
        kt, vt = inputs

        ww = u_f32 + kt
        p = jnp.maximum(pp, ww)
        e1 = jnp.exp(pp - p)
        e2 = jnp.exp(ww - p)
        wkv_t = (e1 * aa + e2 * vt) / (e1 * bb + e2 + eps)

        ww_next = pp + w_f32
        p_next = jnp.maximum(ww_next, kt)
        e1_next = jnp.exp(ww_next - p_next)
        e2_next = jnp.exp(kt - p_next)
        aa_next = e1_next * aa + e2_next * vt
        bb_next = e1_next * bb + e2_next
        pp_next = p_next

        return (aa_next, bb_next, pp_next), wkv_t

    (_, _, _), wkv = lax.scan(step, (init_aa, init_bb, init_pp), (k_f32, v_f32))
    wkv = jnp.transpose(wkv, (1, 0, 2))
    rwkv = jax.nn.sigmoid(r.astype(jnp.float32)) * wkv
    return rwkv.astype(r.dtype)
