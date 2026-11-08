"""RoPE on new keys at absolute positions + per-sequence-offset append of K/V into a contiguous KV cache."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'kv_rope_append',
    'batch': 8,
    'max_seq_len': 8192,
    'num_kv_heads': 8,
    'head_dim': 128,
    'new_tokens': 256,
    'rope_theta': 500000.0,
}


def create_inputs(dtype=jnp.bfloat16):
    B = CONFIG['batch']
    S = CONFIG['max_seq_len']
    H = CONFIG['num_kv_heads']
    D = CONFIG['head_dim']
    T = CONFIG['new_tokens']
    key = jax.random.key(0)
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    k_cache = jax.random.normal(k1, (B, S, H, D), dtype=jnp.float32).astype(dtype)
    v_cache = jax.random.normal(k2, (B, S, H, D), dtype=jnp.float32).astype(dtype)
    new_k = jax.random.normal(k3, (B, T, H, D), dtype=jnp.float32).astype(dtype)
    new_v = jax.random.normal(k4, (B, T, H, D), dtype=jnp.float32).astype(dtype)
    offsets = jax.random.randint(k5, (B,), 0, S - T + 1, dtype=jnp.int32)
    return k_cache, v_cache, new_k, new_v, offsets


def _rope(x, positions, theta):
    # x: [B, T, H, D]; positions: [B, T] int32. Neox rotate-half form, fp32 math.
    D = x.shape[-1]
    half = D // 2
    inv_freq = 1.0 / (theta ** (jnp.arange(0, half, dtype=jnp.float32) * 2.0 / D))
    angles = positions.astype(jnp.float32)[:, :, None] * inv_freq[None, None, :]  # [B, T, half]
    cos = jnp.cos(angles)[:, :, None, :]
    sin = jnp.sin(angles)[:, :, None, :]
    xf = x.astype(jnp.float32)
    x1 = xf[..., :half]
    x2 = xf[..., half:]
    out = jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)
    return out.astype(x.dtype)


def workload(k_cache, v_cache, new_k, new_v, offsets):
    T = new_k.shape[1]
    theta = CONFIG['rope_theta']
    positions = offsets[:, None] + jnp.arange(T, dtype=jnp.int32)[None, :]  # [B, T]
    k_rot = _rope(new_k, positions, theta).astype(k_cache.dtype)
    v_new = new_v.astype(v_cache.dtype)

    def _append(cache, upd, off):
        zero = jnp.zeros((), dtype=jnp.int32)
        return lax.dynamic_update_slice(cache, upd, (off, zero, zero))

    k_out = jax.vmap(_append)(k_cache, k_rot, offsets)
    v_out = jax.vmap(_append)(v_cache, v_new, offsets)
    return k_out, v_out
