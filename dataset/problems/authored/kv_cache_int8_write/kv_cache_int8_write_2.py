"""Per-token per-head absmax int8 quantization of new K/V fused with write into preallocated int8 KV cache + scale cache at per-sequence positions."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'kv_cache_int8_write',
    'batch': 8,
    'num_kv_heads': 8,
    'max_seq_len': 8192,
    'head_dim': 128,
    'new_tokens': 512,
    'qmax': 127.0,
    'eps': 1e-8,
}


def create_inputs(dtype=jnp.float32):
    B = CONFIG['batch']
    H = CONFIG['num_kv_heads']
    S = CONFIG['max_seq_len']
    D = CONFIG['head_dim']
    T = CONFIG['new_tokens']
    key = jax.random.key(0)
    k1, k2, k3, k4, k5, k6, k7 = jax.random.split(key, 7)
    k_new = jax.random.normal(k1, (B, T, H, D), dtype)
    v_new = jax.random.normal(k2, (B, T, H, D), dtype)
    k_cache = jax.random.randint(k3, (B, H, S, D), -127, 128, dtype=jnp.int32).astype(jnp.int8)
    v_cache = jax.random.randint(k4, (B, H, S, D), -127, 128, dtype=jnp.int32).astype(jnp.int8)
    k_scale = jax.random.uniform(k5, (B, H, S), jnp.float32, 1e-3, 2e-2)
    v_scale = jax.random.uniform(k6, (B, H, S), jnp.float32, 1e-3, 2e-2)
    positions = jax.random.randint(k7, (B,), 0, S - T + 1, dtype=jnp.int32)
    return k_new, v_new, k_cache, v_cache, k_scale, v_scale, positions


def _quantize(x):
    # x: [B, T, H, D] -> q: [B, H, T, D] int8, scale: [B, H, T] f32
    qmax = CONFIG['qmax']
    xf = x.astype(jnp.float32)
    absmax = jnp.max(jnp.abs(xf), axis=-1)
    scale = jnp.maximum(absmax, CONFIG['eps']) / qmax
    q = jnp.clip(jnp.round(xf / scale[..., None]), -qmax, qmax).astype(jnp.int8)
    return jnp.transpose(q, (0, 2, 1, 3)), jnp.transpose(scale, (0, 2, 1))


def _write_one(cache, scale_cache, q, s, pos):
    # cache: [H, S, D], scale_cache: [H, S], q: [H, T, D], s: [H, T]
    zero = jnp.zeros((), jnp.int32)
    cache = lax.dynamic_update_slice(cache, q, (zero, pos, zero))
    scale_cache = lax.dynamic_update_slice(scale_cache, s.astype(scale_cache.dtype), (zero, pos))
    return cache, scale_cache


def workload(k_new, v_new, k_cache, v_cache, k_scale, v_scale, positions):
    kq, ks = _quantize(k_new)
    vq, vs = _quantize(v_new)
    write = jax.vmap(_write_one, in_axes=(0, 0, 0, 0, 0))
    pos = positions.astype(jnp.int32)
    k_cache_out, k_scale_out = write(k_cache, k_scale, kq, ks, pos)
    v_cache_out, v_scale_out = write(v_cache, v_scale, vq, vs, pos)
    return k_cache_out, v_cache_out, k_scale_out, v_scale_out
