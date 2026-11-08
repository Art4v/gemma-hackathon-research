"""Sliding-window KV cache ring buffer update and validity mask generation."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'sliding_window_kv_ring_buffer_update',
    'batch_size': 32,
    'window_size': 4096,
    'update_len': 64,
    'num_kv_heads': 8,
    'head_dim': 128,
}

def create_inputs(dtype=jnp.bfloat16):
    k0, k1, k2, k3, k4 = jax.random.split(jax.random.key(0), 5)
    B = CONFIG['batch_size']
    W = CONFIG['window_size']
    S = CONFIG['update_len']
    H = CONFIG['num_kv_heads']
    D = CONFIG['head_dim']

    k_cache = jax.random.normal(k0, (B, W, H, D), dtype=dtype)
    v_cache = jax.random.normal(k1, (B, W, H, D), dtype=dtype)
    k_new = jax.random.normal(k2, (B, S, H, D), dtype=dtype)
    v_new = jax.random.normal(k3, (B, S, H, D), dtype=dtype)
    positions = jax.random.randint(k4, (B,), minval=0, maxval=W * 2, dtype=jnp.int32)

    return k_cache, v_cache, k_new, v_new, positions

def workload(k_cache, v_cache, k_new, v_new, positions):
    B, W, H, D = k_cache.shape
    S = k_new.shape[1]

    step_offsets = jnp.arange(S, dtype=jnp.int32)[None, :]
    write_indices = (positions[:, None] + step_offsets) % W
    batch_indices = jnp.arange(B, dtype=jnp.int32)[:, None]

    updated_k_cache = k_cache.at[batch_indices, write_indices].set(k_new)
    updated_v_cache = v_cache.at[batch_indices, write_indices].set(v_new)

    new_positions = positions + S
    slot_indices = jnp.arange(W, dtype=jnp.int32)[None, :]
    valid_mask = slot_indices < new_positions[:, None]

    return updated_k_cache, updated_v_cache, valid_mask, new_positions
