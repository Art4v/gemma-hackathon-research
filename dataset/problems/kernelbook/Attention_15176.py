"""Attention_15176: 1x1 conv QKV projections -> QK^T -> softmax -> AV -> 1x1 conv output projection."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Attention_15176',
    'batch': 8,
    'C_in_q': 256,
    'C_in_kv': 256,
    'C_qk': 256,
    'C_v': 256,
    'H': 64,
    'W': 64,
    'n_heads': 1,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    ks = jax.random.split(jax.random.key(0), 10)
    x = jax.random.uniform(ks[0], (c['batch'], c['C_in_q'], c['H'], c['W']), dtype)
    y = jax.random.uniform(ks[1], (c['batch'], c['C_in_kv'], c['H'], c['W']), dtype)
    wq = jax.random.normal(ks[2], (c['C_in_q'], c['C_qk']), dtype) * 0.02
    bq = jax.random.normal(ks[3], (c['C_qk'],), dtype) * 0.02
    wk = jax.random.normal(ks[4], (c['C_in_kv'], c['C_qk']), dtype) * 0.02
    bk = jax.random.normal(ks[5], (c['C_qk'],), dtype) * 0.02
    wv = jax.random.normal(ks[6], (c['C_in_kv'], c['C_v']), dtype) * 0.02
    bv = jax.random.normal(ks[7], (c['C_v'],), dtype) * 0.02
    wo = jax.random.normal(ks[8], (c['C_v'], c['C_v']), dtype) * 0.02
    bo = jax.random.normal(ks[9], (c['C_v'],), dtype) * 0.02
    return x, y, wq, bq, wk, bk, wv, bv, wo, bo


def _proj(flat, w, b):
    # 1x1 Conv1d: (B, C_in, N) -> (B, C_out, N)
    return jnp.einsum('bcn,cd->bdn', flat, w) + b[None, :, None]


def _split_heads(t, n_heads):
    b, ch, n = t.shape
    return t.reshape(b, n_heads, ch // n_heads, n)


def workload(x, y, wq, bq, wk, bk, wv, bv, wo, bo):
    B, C, H, W = x.shape
    n_heads = CONFIG['n_heads']
    flat_x = x.reshape(B, C, H * W)
    flat_y = y.reshape(y.shape[0], y.shape[1], -1)
    q = _split_heads(_proj(flat_x, wq, bq), n_heads)
    k = _split_heads(_proj(flat_y, wk, bk), n_heads)
    v = _split_heads(_proj(flat_y, wv, bv), n_heads)
    score = jnp.einsum('bhcq,bhck->bhqk', q, k)
    attn_w = jax.nn.softmax(score, axis=-1)
    attn_out = jnp.einsum('bhqk,bhck->bhcq', attn_w, v).reshape(B, C, H, W)
    # 1x1 Conv2d output projection
    return jnp.einsum('bchw,cd->bdhw', attn_out, wo) + bo[None, :, None, None]
