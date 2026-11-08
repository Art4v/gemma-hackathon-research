"""Sliding-window KV cache ring buffer update with validity mask generation."""

import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'sliding_window_kv_ring_buffer_update',
    'batch_size': 32,
    'window_size': 4096,
    'update_len': 128,
    'num_kv_heads': 8,
    'head_dim': 128,
    'max_seq_len': 16384,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)

    B = CONFIG['batch_size']
    W = CONFIG['window_size']
    U = CONFIG['update_len']
    H = CONFIG['num_kv_heads']
    D = CONFIG['head_dim']
    M = CONFIG['max_seq_len']

    k_cache = jax.random.normal(k1, (B, W, H, D), dtype=dtype)
    v_cache = jax.random.normal(k2, (B, W, H, D), dtype=dtype)
    new_k = jax.random.normal(k3, (B, U, H, D), dtype=dtype)
    new_v = jax.random.normal(k4, (B, U, H, D), dtype=dtype)
    write_positions = jax.random.randint(k5, (B,), minval=0, maxval=M, dtype=jnp.int32)

    return k_cache, v_cache, new_k, new_v, write_positions


def workload(k_cache, v_cache, new_k, new_v, write_positions):
    W = CONFIG['window_size']
    U = CONFIG['update_len']

    def update_single(k_c, v_c, nk, nv, pos):
        offsets = jnp.arange(U, dtype=jnp.int32)
        ring_indices = (pos + offsets) % W
        updated_k = k_c.at[ring_indices].set(nk)
        updated_v = v_c.at[ring_indices].set(nv)
        new_len = pos + U
        slot_indices = jnp.arange(W, dtype=jnp.int32)
        valid_mask = slot_indices < jnp.minimum(new_len, W)
        return updated_k, updated_v, valid_mask

    return jax.vmap(update_single)(k_cache, v_cache, new_k, new_v, write_positions)
