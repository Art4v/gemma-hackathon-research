"""Paged KV cache scatter update (reshape_and_cache) by slot mapping."""

import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'reshape_and_cache',
    'num_tokens': 16384,
    'num_blocks': 4096,
    'block_size': 16,
    'num_kv_heads': 8,
    'head_dim': 128,
}


def create_inputs(dtype=jnp.bfloat16):
    num_tokens = CONFIG['num_tokens']
    num_blocks = CONFIG['num_blocks']
    block_size = CONFIG['block_size']
    num_kv_heads = CONFIG['num_kv_heads']
    head_dim = CONFIG['head_dim']

    k0, k1, k2, k3, k4 = jax.random.split(jax.random.key(0), 5)

    key = jax.random.normal(k0, (num_tokens, num_kv_heads, head_dim), dtype=dtype)
    value = jax.random.normal(k1, (num_tokens, num_kv_heads, head_dim), dtype=dtype)

    key_cache = jax.random.normal(
        k2, (num_blocks, block_size, num_kv_heads, head_dim), dtype=dtype
    )
    value_cache = jax.random.normal(
        k3, (num_blocks, block_size, num_kv_heads, head_dim), dtype=dtype
    )

    total_slots = num_blocks * block_size
    all_slots = jax.random.permutation(k4, jnp.arange(total_slots, dtype=jnp.int32))
    slot_mapping = all_slots[:num_tokens]

    return key, value, key_cache, value_cache, slot_mapping


def workload(key, value, key_cache, value_cache, slot_mapping):
    num_blocks, block_size, num_kv_heads, head_dim = key_cache.shape
    total_slots = num_blocks * block_size

    flat_k_cache = key_cache.reshape(total_slots, num_kv_heads, head_dim)
    flat_v_cache = value_cache.reshape(total_slots, num_kv_heads, head_dim)

    safe_slots = jnp.where(slot_mapping >= 0, slot_mapping, total_slots)

    updated_flat_k = flat_k_cache.at[safe_slots].set(
        key, mode='drop', unique_indices=True
    )
    updated_flat_v = flat_v_cache.at[safe_slots].set(
        value, mode='drop', unique_indices=True
    )

    new_key_cache = updated_flat_k.reshape(
        num_blocks, block_size, num_kv_heads, head_dim
    )
    new_value_cache = updated_flat_v.reshape(
        num_blocks, block_size, num_kv_heads, head_dim
    )

    return new_key_cache, new_value_cache
