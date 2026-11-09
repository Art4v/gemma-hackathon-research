"""[m1 M1_batch variant of authored/kv_cache_ring_buffer/sliding_window_kv_ring_update] The ring-buffer K/V scatter now updates a stack of L local-attention layers' caches at once, sharing one per-sequence slot-position table and validity mask."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'sliding_window_kv_ring_update_m1_batch',
    'num_layers': 4,
    'batch': 8,
    'window': 1024,
    'num_kv_heads': 8,
    'head_dim': 128,
    'num_new_tokens': 256,
    'max_start_windows': 3,
}


def _initial_slot_pos(start_pos, window):
    # Largest absolute position p < start_pos with p % window == w, else -1.
    w = jnp.arange(window, dtype=jnp.int32)[None, :]
    last = start_pos[:, None] - 1
    d = last - w
    p = w + window * jnp.floor_divide(d, window)
    return jnp.where(d >= 0, p, -1).astype(jnp.int32)


def create_inputs(dtype=jnp.float32):
    L = CONFIG['num_layers']
    B = CONFIG['batch']
    W = CONFIG['window']
    H = CONFIG['num_kv_heads']
    D = CONFIG['head_dim']
    T = CONFIG['num_new_tokens']
    assert T <= W
    key = jax.random.key(0)
    k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
    k_cache = jax.random.normal(k1, (L, B, W, H, D), dtype=dtype)
    v_cache = jax.random.normal(k2, (L, B, W, H, D), dtype=dtype)
    k_new = jax.random.normal(k3, (L, B, T, H, D), dtype=dtype)
    v_new = jax.random.normal(k4, (L, B, T, H, D), dtype=dtype)
    start_pos = jax.random.randint(k5, (B,), 0, CONFIG['max_start_windows'] * W, dtype=jnp.int32)
    num_new = jax.random.randint(k6, (B,), 1, T + 1, dtype=jnp.int32)
    slot_pos = _initial_slot_pos(start_pos, W)
    # Stale slots (never written) hold garbage K/V; zero them for sanity.
    filled = (slot_pos >= 0)[None, :, :, None, None]
    k_cache = jnp.where(filled, k_cache, jnp.zeros_like(k_cache))
    v_cache = jnp.where(filled, v_cache, jnp.zeros_like(v_cache))
    return k_cache, v_cache, k_new, v_new, slot_pos, start_pos, num_new


def workload(k_cache, v_cache, k_new, v_new, slot_pos, start_pos, num_new):
    # Positions are per sequence, so every layer writes the same slots.
    B, W = slot_pos.shape
    T = k_new.shape[2]
    t = jnp.arange(T, dtype=jnp.int32)[None, :]                 # [1, T]
    abs_pos = start_pos[:, None] + t                            # [B, T]
    live = t < num_new[:, None]                                 # [B, T]
    slots = jnp.where(live, jnp.mod(abs_pos, W), W)             # W => dropped
    b_idx = jnp.broadcast_to(jnp.arange(B, dtype=jnp.int32)[:, None], (B, T))

    k_out = k_cache.at[:, b_idx, slots].set(k_new.astype(k_cache.dtype), mode='drop')
    v_out = v_cache.at[:, b_idx, slots].set(v_new.astype(v_cache.dtype), mode='drop')
    pos_out = slot_pos.at[b_idx, slots].set(abs_pos.astype(slot_pos.dtype), mode='drop')

    end = (start_pos + num_new)[:, None]                        # [B, 1]
    valid = (pos_out >= 0) & (pos_out >= end - W) & (pos_out < end)
    return k_out, v_out, pos_out, valid
