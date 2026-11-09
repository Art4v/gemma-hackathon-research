"""[m1 M1_batch variant of authored/kv_cache_ring_buffer/sliding_window_kv_ring_update_2] The ring-buffer K/V scatter at (pos mod window) now updates a stack of L local-attention layers' caches at once, sharing one per-sequence slot-position table and validity mask."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'sliding_window_kv_ring_update_m1_batch',
    'num_layers': 4,
    'batch': 8,
    'window': 1024,
    'num_kv_heads': 8,
    'head_dim': 128,
    'new_tokens': 512,
    'max_start_multiple': 3,  # start positions drawn from [0, max_start_multiple * window)
}
assert CONFIG['new_tokens'] <= CONFIG['window']


def create_inputs(dtype=jnp.bfloat16):
    L = CONFIG['num_layers']
    B = CONFIG['batch']
    W = CONFIG['window']
    H = CONFIG['num_kv_heads']
    D = CONFIG['head_dim']
    T = CONFIG['new_tokens']
    key = jax.random.key(0)
    k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
    k_cache = jax.random.normal(k1, (L, B, W, H, D), dtype=jnp.float32).astype(dtype)
    v_cache = jax.random.normal(k2, (L, B, W, H, D), dtype=jnp.float32).astype(dtype)
    new_k = jax.random.normal(k3, (L, B, T, H, D), dtype=jnp.float32).astype(dtype)
    new_v = jax.random.normal(k4, (L, B, T, H, D), dtype=jnp.float32).astype(dtype)
    start = jax.random.randint(k5, (B,), 0, CONFIG['max_start_multiple'] * W, dtype=jnp.int32)
    n_new = jax.random.randint(k6, (B,), 1, T + 1, dtype=jnp.int32)
    # Consistent slot position table: slot j holds the latest position p < start with p % W == j, or -1.
    j = jnp.arange(W, dtype=jnp.int32)[None, :]
    last = start[:, None] - 1
    p = last - jnp.mod(last - j, W)
    slot_pos = jnp.where(p >= 0, p, -1).astype(jnp.int32)
    return k_cache, v_cache, new_k, new_v, start, n_new, slot_pos


def workload(k_cache, v_cache, new_k, new_v, start, n_new, slot_pos):
    # Positions are per sequence, so every layer writes the same slots.
    B, W = k_cache.shape[1], k_cache.shape[2]
    T = new_k.shape[2]
    t = jnp.arange(T, dtype=jnp.int32)[None, :]                 # [1, T]
    abs_pos = start[:, None] + t                                 # [B, T]
    write = t < n_new[:, None]                                   # [B, T] ragged validity
    slots = jnp.where(write, jnp.mod(abs_pos, W), W)             # out-of-range index -> dropped
    b_idx = jnp.broadcast_to(jnp.arange(B, dtype=jnp.int32)[:, None], (B, T))

    k_out = k_cache.at[:, b_idx, slots].set(new_k.astype(k_cache.dtype), mode='drop')
    v_out = v_cache.at[:, b_idx, slots].set(new_v.astype(v_cache.dtype), mode='drop')
    pos_out = slot_pos.at[b_idx, slots].set(abs_pos.astype(slot_pos.dtype), mode='drop')

    end = (start + n_new)[:, None]                               # [B, 1] next position to write
    valid = (pos_out >= 0) & (pos_out >= end - W) & (pos_out < end)
    return k_out, v_out, pos_out, valid
