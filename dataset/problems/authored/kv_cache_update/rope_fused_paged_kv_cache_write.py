"""Fused NeoX RoPE on new K + scatter of K/V into a paged (block_size-slot) KV cache via slot mapping, skipping padded (-1) slots."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'rope_fused_paged_kv_cache_write',
    'num_tokens': 8192,
    'num_pad_tokens': 256,
    'num_kv_heads': 8,
    'head_dim': 128,
    'num_blocks': 4096,
    'block_size': 16,
    'max_position': 131072,
    'rope_theta': 500000.0,
}


def create_inputs(dtype=jnp.bfloat16):
    T = CONFIG['num_tokens']
    H = CONFIG['num_kv_heads']
    D = CONFIG['head_dim']
    NB = CONFIG['num_blocks']
    BS = CONFIG['block_size']
    num_slots = NB * BS
    key = jax.random.key(0)
    k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
    k_new = jax.random.normal(k1, (T, H, D), dtype)
    v_new = jax.random.normal(k2, (T, H, D), dtype)
    positions = jax.random.randint(k3, (T,), 0, CONFIG['max_position'], dtype=jnp.int32)
    slots = jax.random.permutation(k4, num_slots)[:T].astype(jnp.int32)
    pad = CONFIG['num_pad_tokens']
    is_pad = jnp.arange(T) >= (T - pad)
    slot_mapping = jnp.where(is_pad, jnp.int32(-1), slots)
    k_cache = jax.random.normal(k5, (NB, BS, H, D), dtype)
    v_cache = jax.random.normal(k6, (NB, BS, H, D), dtype)
    return k_new, v_new, positions, slot_mapping, k_cache, v_cache


def workload(k_new, v_new, positions, slot_mapping, k_cache, v_cache):
    T, H, D = k_new.shape
    NB, BS = k_cache.shape[0], k_cache.shape[1]
    num_slots = NB * BS
    half = D // 2

    # RoPE (NeoX rotate-half) in float32.
    inv_freq = 1.0 / (CONFIG['rope_theta'] ** (jnp.arange(half, dtype=jnp.float32) * 2.0 / D))
    angles = positions.astype(jnp.float32)[:, None] * inv_freq[None, :]  # [T, half]
    cos = jnp.cos(angles)[:, None, :]
    sin = jnp.sin(angles)[:, None, :]
    kf = k_new.astype(jnp.float32)
    k1 = kf[..., :half]
    k2 = kf[..., half:]
    k_rot = jnp.concatenate([k1 * cos - k2 * sin, k2 * cos + k1 * sin], axis=-1)
    k_rot = k_rot.astype(k_cache.dtype)
    v_w = v_new.astype(v_cache.dtype)

    # Padded tokens (slot < 0) -> out-of-range index, dropped by scatter.
    idx = jnp.where(slot_mapping < 0, num_slots, slot_mapping)

    k_flat = k_cache.reshape(num_slots, H, D)
    v_flat = v_cache.reshape(num_slots, H, D)
    k_flat = k_flat.at[idx].set(k_rot, mode='drop', unique_indices=True)
    v_flat = v_flat.at[idx].set(v_w, mode='drop', unique_indices=True)
    return k_flat.reshape(k_cache.shape), v_flat.reshape(v_cache.shape)
