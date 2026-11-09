"""[m1 M1_batch variant of authored/kv_cache_ring_buffer/sliding_window_kv_ring_buffer_update] Sliding-window KV ring-buffer update for a leading axis of 4 layers, each with its own K/V ring and slot-position table, sharing the per-sequence decode position, with batch cut from 16 to 4."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'sliding_window_kv_ring_buffer_update_m1_batch',
    'num_layers': 4,
    'batch': 4,
    'window': 4096,
    'num_kv_heads': 8,
    'head_dim': 128,
    'num_new_tokens': 512,
    'max_pos_windows': 3,
}


def create_inputs(dtype=jnp.float32):
    L = CONFIG['num_layers']
    B = CONFIG['batch']
    W = CONFIG['window']
    H = CONFIG['num_kv_heads']
    D = CONFIG['head_dim']
    T = CONFIG['num_new_tokens']
    key = jax.random.key(0)
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    k_cache = jax.random.normal(k1, (L, B, W, H, D), dtype)
    v_cache = jax.random.normal(k2, (L, B, W, H, D), dtype)
    k_new = jax.random.normal(k3, (L, B, T, H, D), dtype)
    v_new = jax.random.normal(k4, (L, B, T, H, D), dtype)
    # One decode position per sequence, shared by every layer.
    pos = jax.random.randint(k5, (B,), 0, CONFIG['max_pos_windows'] * W, dtype=jnp.int32)
    # slot_pos[l, b, s] = largest absolute position p < pos[b] with p % W == s, else -1.
    s = jnp.arange(W, dtype=jnp.int32)[None, :]
    last = pos[:, None] - 1
    p = s + W * jnp.floor_divide(last - s, W)
    slot_pos = jnp.where(last >= s, p, -1).astype(jnp.int32)
    # Zero out cache contents in never-written slots (deterministic, realistic).
    filled = (slot_pos >= 0)[:, :, None, None]
    k_cache = jnp.where(filled, k_cache, jnp.zeros((), dtype))
    v_cache = jnp.where(filled, v_cache, jnp.zeros((), dtype))
    # Each layer's ring buffer carries its own copy of the slot-position table.
    slot_pos = jnp.broadcast_to(slot_pos, (L, B, W))
    return k_cache, v_cache, k_new, v_new, pos, slot_pos


def _layer(k_cache, v_cache, k_new, v_new, pos, slot_pos):
    B, W = k_cache.shape[0], k_cache.shape[1]
    T = k_new.shape[1]
    t = jnp.arange(T, dtype=jnp.int32)
    abs_pos = pos[:, None].astype(jnp.int32) + t[None, :]          # [B, T]
    slot = jnp.mod(abs_pos, W)                                      # [B, T], unique per row since T <= W
    b_idx = jnp.broadcast_to(jnp.arange(B, dtype=jnp.int32)[:, None], (B, T))
    k_out = k_cache.at[b_idx, slot].set(k_new.astype(k_cache.dtype), unique_indices=True)
    v_out = v_cache.at[b_idx, slot].set(v_new.astype(v_cache.dtype), unique_indices=True)
    slot_pos_out = slot_pos.at[b_idx, slot].set(abs_pos, unique_indices=True)
    new_len = pos[:, None].astype(jnp.int32) + T                    # [B, 1]
    valid = (slot_pos_out >= 0) & (slot_pos_out >= new_len - W) & (slot_pos_out < new_len)
    return k_out, v_out, slot_pos_out, valid


def workload(k_cache, v_cache, k_new, v_new, pos, slot_pos):
    # Leading layer axis on caches, new K/V and slot tables; pos is shared across layers.
    return jax.vmap(_layer, in_axes=(0, 0, 0, 0, None, 0))(
        k_cache, v_cache, k_new, v_new, pos, slot_pos)
