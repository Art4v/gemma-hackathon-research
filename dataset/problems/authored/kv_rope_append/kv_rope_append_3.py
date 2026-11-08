"""RoPE(K_new at absolute positions offsets[b]+t) -> append K/V into contiguous KV cache at per-sequence offsets."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'kv_rope_append',
    'batch': 8,
    'num_kv_heads': 8,
    'max_seq_len': 4096,
    'head_dim': 128,
    'new_tokens': 256,
    'rope_theta': 500000.0,
}


def create_inputs(dtype=jnp.float32):
    B = CONFIG['batch']
    H = CONFIG['num_kv_heads']
    S = CONFIG['max_seq_len']
    D = CONFIG['head_dim']
    T = CONFIG['new_tokens']
    key = jax.random.key(0)
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    k_new = jax.random.normal(k1, (B, H, T, D), dtype=dtype)
    v_new = jax.random.normal(k2, (B, H, T, D), dtype=dtype)
    k_cache = jax.random.normal(k3, (B, H, S, D), dtype=dtype)
    v_cache = jax.random.normal(k4, (B, H, S, D), dtype=dtype)
    offsets = jax.random.randint(k5, (B,), 0, S - T + 1, dtype=jnp.int32)
    return k_new, v_new, k_cache, v_cache, offsets


def _rope(x, positions, theta):
    # x: [H, T, D]; positions: [T] int32. Half-split (rotate_half) RoPE in float32.
    D = x.shape[-1]
    half = D // 2
    inv_freq = 1.0 / (theta ** (jnp.arange(0, half, dtype=jnp.float32) * 2.0 / D))
    angles = positions.astype(jnp.float32)[:, None] * inv_freq[None, :]  # [T, half]
    cos = jnp.cos(angles)[None, :, :]
    sin = jnp.sin(angles)[None, :, :]
    xf = x.astype(jnp.float32)
    x1 = xf[..., :half]
    x2 = xf[..., half:]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return jnp.concatenate([o1, o2], axis=-1).astype(x.dtype)


def workload(k_new, v_new, k_cache, v_cache, offsets):
    theta = CONFIG['rope_theta']
    T = k_new.shape[2]

    def per_seq(kn, vn, kc, vc, off):
        positions = off + jnp.arange(T, dtype=jnp.int32)
        k_rot = _rope(kn, positions, theta).astype(kc.dtype)
        zero = jnp.zeros((), dtype=jnp.int32)
        kc = lax.dynamic_update_slice(kc, k_rot, (zero, off, zero))
        vc = lax.dynamic_update_slice(vc, vn.astype(vc.dtype), (zero, off, zero))
        return kc, vc

    new_k_cache, new_v_cache = jax.vmap(per_seq)(k_new, v_new, k_cache, v_cache, offsets)
    return new_k_cache, new_v_cache
