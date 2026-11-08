"""Rotary embedding on new K at absolute positions + append of K/V into contiguous KV cache at per-sequence offsets."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'kv_rope_append',
    'batch': 16,
    'max_seq_len': 4096,
    'num_kv_heads': 8,
    'head_dim': 128,
    'new_tokens': 16,
    'rope_theta': 500000.0,
}


def create_inputs(dtype=jnp.float32):
    B = CONFIG['batch']
    S = CONFIG['max_seq_len']
    H = CONFIG['num_kv_heads']
    D = CONFIG['head_dim']
    T = CONFIG['new_tokens']
    k0, k1, k2, k3, k4 = jax.random.split(jax.random.key(0), 5)
    k_cache = jax.random.normal(k0, (B, S, H, D), dtype)
    v_cache = jax.random.normal(k1, (B, S, H, D), dtype)
    k_new = jax.random.normal(k2, (B, T, H, D), dtype)
    v_new = jax.random.normal(k3, (B, T, H, D), dtype)
    offsets = jax.random.randint(k4, (B,), 0, S - T + 1, dtype=jnp.int32)
    return k_cache, v_cache, k_new, v_new, offsets


def _rope(k, positions):
    # k: [B, T, H, D]; positions: [B, T] int32. Rotate-half (Llama/HF) style, f32 math.
    D = k.shape[-1]
    half = D // 2
    inv_freq = 1.0 / (CONFIG['rope_theta'] ** (jnp.arange(0, half, dtype=jnp.float32) * 2.0 / D))
    angles = positions.astype(jnp.float32)[..., None] * inv_freq  # [B, T, half]
    cos = jnp.cos(angles)[:, :, None, :]  # [B, T, 1, half]
    sin = jnp.sin(angles)[:, :, None, :]
    kf = k.astype(jnp.float32)
    k1 = kf[..., :half]
    k2 = kf[..., half:]
    out = jnp.concatenate([k1 * cos - k2 * sin, k2 * cos + k1 * sin], axis=-1)
    return out.astype(k.dtype)


def workload(k_cache, v_cache, k_new, v_new, offsets):
    T = k_new.shape[1]
    offsets = offsets.astype(jnp.int32)
    positions = offsets[:, None] + jnp.arange(T, dtype=jnp.int32)[None, :]  # [B, T]
    k_rot = _rope(k_new, positions).astype(k_cache.dtype)
    v_in = v_new.astype(v_cache.dtype)

    def _write(cache_b, new_b, off):
        zero = jnp.zeros((), jnp.int32)
        return lax.dynamic_update_slice(cache_b, new_b, (off, zero, zero))

    k_out = jax.vmap(_write)(k_cache, k_rot, offsets)
    v_out = jax.vmap(_write)(v_cache, v_in, offsets)
    return k_out, v_out
