"""[m1 M1_batch variant of authored/kv_cache_int8_write/kv_cache_int8_write] Per-token per-head absmax int8 quantize-and-write of new K/V for a leading axis of 2 layers at once, sharing the per-sequence write positions, with batch cut from 16 to 8."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'kv_cache_int8_write_m1_batch',
    'num_layers': 2,
    'batch': 8,
    'num_kv_heads': 8,
    'head_dim': 128,
    'max_seq_len': 8192,
    'new_tokens': 16,
    'qmax': 127.0,
    'eps': 1e-8,
}


def create_inputs(dtype=jnp.float32):
    L = CONFIG['num_layers']
    B = CONFIG['batch']
    H = CONFIG['num_kv_heads']
    D = CONFIG['head_dim']
    S = CONFIG['max_seq_len']
    T = CONFIG['new_tokens']
    key = jax.random.key(0)
    k1, k2, k3, k4, k5, k6, k7 = jax.random.split(key, 7)
    new_k = jax.random.normal(k1, (L, B, T, H, D), dtype)
    new_v = jax.random.normal(k2, (L, B, T, H, D), dtype)
    k_cache = jax.random.randint(k3, (L, B, H, S, D), -127, 128, dtype=jnp.int8)
    v_cache = jax.random.randint(k4, (L, B, H, S, D), -127, 128, dtype=jnp.int8)
    k_scale = jax.random.uniform(k5, (L, B, H, S), jnp.float32) * 0.05
    v_scale = jax.random.uniform(k6, (L, B, H, S), jnp.float32) * 0.05
    # One write position per sequence, shared by every layer.
    positions = jax.random.randint(k7, (B,), 0, S - T + 1, dtype=jnp.int32)
    return new_k, new_v, k_cache, v_cache, k_scale, v_scale, positions


def _quantize(x):
    # x: [B, T, H, D] -> q: [B, H, T, D] int8, scale: [B, H, T] float32
    qmax = CONFIG['qmax']
    xf = jnp.transpose(x.astype(jnp.float32), (0, 2, 1, 3))
    absmax = jnp.max(jnp.abs(xf), axis=-1)
    scale = jnp.maximum(absmax / qmax, CONFIG['eps'])
    q = jnp.clip(jnp.round(xf / scale[..., None]), -qmax, qmax).astype(jnp.int8)
    return q, scale


def _write_one(cache, scale_cache, q, s, pos):
    # cache: [H, S, D], scale_cache: [H, S], q: [H, T, D], s: [H, T]
    zero = jnp.zeros((), jnp.int32)
    cache = lax.dynamic_update_slice(cache, q, (zero, pos, zero))
    scale_cache = lax.dynamic_update_slice(scale_cache, s, (zero, pos))
    return cache, scale_cache


def _layer(new_k, new_v, k_cache, v_cache, k_scale, v_scale, positions):
    qk, sk = _quantize(new_k)
    qv, sv = _quantize(new_v)
    pos = positions.astype(jnp.int32)
    write = jax.vmap(_write_one)
    k_cache_out, k_scale_out = write(k_cache, k_scale, qk, sk, pos)
    v_cache_out, v_scale_out = write(v_cache, v_scale, qv, sv, pos)
    return k_cache_out, v_cache_out, k_scale_out, v_scale_out


def workload(new_k, new_v, k_cache, v_cache, k_scale, v_scale, positions):
    # Leading layer axis on every K/V tensor; positions are shared across layers.
    return jax.vmap(_layer, in_axes=(0, 0, 0, 0, 0, 0, None))(
        new_k, new_v, k_cache, v_cache, k_scale, v_scale, positions)
