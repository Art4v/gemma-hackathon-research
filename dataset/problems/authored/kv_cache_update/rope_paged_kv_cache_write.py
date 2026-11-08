"""Fused NeoX RoPE on new keys + padding-aware paged KV cache write (reshape_and_cache)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'rope_paged_kv_cache_write',
    'num_tokens': 8192,
    'num_padding_tokens': 512,
    'num_kv_heads': 8,
    'head_dim': 128,
    'num_blocks': 4096,
    'block_size': 16,
    'max_position': 32768,
    'rope_theta': 10000.0,
}


def create_inputs(dtype=jnp.bfloat16):
    T = CONFIG['num_tokens']
    P = CONFIG['num_padding_tokens']
    H = CONFIG['num_kv_heads']
    D = CONFIG['head_dim']
    NB = CONFIG['num_blocks']
    BS = CONFIG['block_size']
    total_slots = NB * BS
    key = jax.random.key(0)
    k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
    new_k = jax.random.normal(k1, (T, H, D), dtype=dtype)
    new_v = jax.random.normal(k2, (T, H, D), dtype=dtype)
    positions = jax.random.randint(k3, (T,), 0, CONFIG['max_position'], dtype=jnp.int32)
    # Unique destination slots; trailing tokens are padding (slot -1, must be skipped).
    perm = jax.random.permutation(k4, total_slots)[:T].astype(jnp.int32)
    is_pad = jnp.arange(T) >= (T - P)
    slot_mapping = jnp.where(is_pad, jnp.int32(-1), perm)
    k_cache = jax.random.normal(k5, (NB, BS, H, D), dtype=dtype)
    v_cache = jax.random.normal(k6, (NB, BS, H, D), dtype=dtype)
    return new_k, new_v, positions, slot_mapping, k_cache, v_cache


def _apply_neox_rope(x, positions, theta):
    # x: [T, H, D]; positions: [T] int32. Compute in float32.
    D = x.shape[-1]
    half = D // 2
    inv_freq = 1.0 / (theta ** (jnp.arange(0, half, dtype=jnp.float32) * 2.0 / D))
    freqs = positions.astype(jnp.float32)[:, None] * inv_freq[None, :]  # [T, half]
    cos = jnp.cos(freqs)[:, None, :]
    sin = jnp.sin(freqs)[:, None, :]
    xf = x.astype(jnp.float32)
    x1 = xf[..., :half]
    x2 = xf[..., half:]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return jnp.concatenate([o1, o2], axis=-1).astype(x.dtype)


def workload(new_k, new_v, positions, slot_mapping, k_cache, v_cache):
    NB, BS, H, D = k_cache.shape
    total_slots = NB * BS
    k_rot = _apply_neox_rope(new_k, positions, CONFIG['rope_theta'])
    # Padding tokens (slot < 0) are redirected out of range and dropped.
    slots = jnp.where(slot_mapping < 0, jnp.int32(total_slots), slot_mapping)
    k_flat = k_cache.reshape(total_slots, H, D)
    v_flat = v_cache.reshape(total_slots, H, D)
    k_flat = k_flat.at[slots].set(k_rot.astype(k_cache.dtype), mode='drop')
    v_flat = v_flat.at[slots].set(new_v.astype(v_cache.dtype), mode='drop')
    return k_flat.reshape(NB, BS, H, D), v_flat.reshape(NB, BS, H, D)
