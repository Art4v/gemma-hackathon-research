"""Sliding-window KV cache ring buffer update with per-sequence chunk lengths and validity mask."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'sliding_window_kv_ring_buffer_update',
    'batch_size': 32,
    'window_size': 4096,
    'update_len_max': 128,
    'num_kv_heads': 8,
    'head_dim': 128,
}

def create_inputs(dtype=jnp.bfloat16):
    B = CONFIG['batch_size']
    W = CONFIG['window_size']
    U = CONFIG['update_len_max']
    H = CONFIG['num_kv_heads']
    D = CONFIG['head_dim']

    k0, k1, k2, k3, k4, k5 = jax.random.split(jax.random.key(0), 6)
    k_cache = jax.random.normal(k0, (B, W, H, D), dtype=dtype)
    v_cache = jax.random.normal(k1, (B, W, H, D), dtype=dtype)
    k_new = jax.random.normal(k2, (B, U, H, D), dtype=dtype)
    v_new = jax.random.normal(k3, (B, U, H, D), dtype=dtype)

    cache_pos = jax.random.randint(k4, (B,), minval=0, maxval=W * 2, dtype=jnp.int32)
    update_len = jax.random.randint(k5, (B,), minval=1, maxval=U + 1, dtype=jnp.int32)

    return k_cache, v_cache, k_new, v_new, cache_pos, update_len

def workload(k_cache, v_cache, k_new, v_new, cache_pos, update_len):
    W = CONFIG['window_size']
    U = CONFIG['update_len_max']

    def update_single(kc, vc, kn, vn, pos, ulen):
        steps = jnp.arange(U, dtype=jnp.int32)
        slots = (pos + steps) % W
        valid_step = steps < ulen

        cur_k = kc[slots]
        cur_v = vc[slots]

        write_k = jnp.where(valid_step[:, None, None], kn, cur_k)
        write_v = jnp.where(valid_step[:, None, None], vn, cur_v)

        kc_out = kc.at[slots].set(write_k)
        vc_out = vc.at[slots].set(write_v)

        new_total_pos = pos + ulen
        w_idx = jnp.arange(W, dtype=jnp.int32)
        valid_mask = w_idx < jnp.minimum(new_total_pos, W)
        return kc_out, vc_out, valid_mask

    return jax.vmap(update_single)(k_cache, v_cache, k_new, v_new, cache_pos, update_len)
