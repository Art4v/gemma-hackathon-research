"""Paged KV cache scatter update (reshape_and_cache) by slot mapping."""

import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'reshape_and_cache',
    'num_tokens': 4096,
    'num_kv_heads': 8,
    'head_dim': 128,
    'num_blocks': 8192,
    'block_size': 16,
    'valid_tokens': 3840,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)

    num_tokens = CONFIG['num_tokens']
    num_kv_heads = CONFIG['num_kv_heads']
    head_dim = CONFIG['head_dim']
    num_blocks = CONFIG['num_blocks']
    block_size = CONFIG['block_size']
    valid_tokens = CONFIG['valid_tokens']
    total_slots = num_blocks * block_size

    new_k = jax.random.normal(k1, (num_tokens, num_kv_heads, head_dim), dtype=dtype)
    new_v = jax.random.normal(k2, (num_tokens, num_kv_heads, head_dim), dtype=dtype)
    k_cache = jax.random.normal(k3, (num_blocks, block_size, num_kv_heads, head_dim), dtype=dtype)
    v_cache = jax.random.normal(k4, (num_blocks, block_size, num_kv_heads, head_dim), dtype=dtype)

    perm = jax.random.permutation(k5, total_slots)[:num_tokens].astype(jnp.int32)
    token_idx = jnp.arange(num_tokens, dtype=jnp.int32)
    slot_mapping = jnp.where(token_idx < valid_tokens, perm, jnp.int32(-1))

    return new_k, new_v, k_cache, v_cache, slot_mapping


def workload(new_k, new_v, k_cache, v_cache, slot_mapping):
    num_blocks, block_size, num_kv_heads, head_dim = k_cache.shape
    total_slots = num_blocks * block_size

    k_cache_flat = k_cache.reshape(total_slots, num_kv_heads, head_dim)
    v_cache_flat = v_cache.reshape(total_slots, num_kv_heads, head_dim)

    safe_slots = jnp.where(slot_mapping >= 0, slot_mapping, total_slots)

    updated_k_flat = k_cache_flat.at[safe_slots].set(new_k, mode='drop')
    updated_v_flat = v_cache_flat.at[safe_slots].set(new_v, mode='drop')

    updated_k_cache = updated_k_flat.reshape(num_blocks, block_size, num_kv_heads, head_dim)
    updated_v_cache = updated_v_flat.reshape(num_blocks, block_size, num_kv_heads, head_dim)

    return updated_k_cache, updated_v_cache
