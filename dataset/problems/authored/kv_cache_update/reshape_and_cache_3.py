"""Paged KV cache scatter update via slot mapping (reshape_and_cache)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'reshape_and_cache',
    'num_tokens': 4096,
    'num_kv_heads': 8,
    'head_dim': 128,
    'num_blocks': 4096,
    'block_size': 16,
}


def create_inputs(dtype=jnp.bfloat16):
    k1, k2, k3, k4, k5 = jax.random.split(jax.random.key(0), 5)
    num_tokens = CONFIG['num_tokens']
    num_kv_heads = CONFIG['num_kv_heads']
    head_dim = CONFIG['head_dim']
    num_blocks = CONFIG['num_blocks']
    block_size = CONFIG['block_size']

    key = jax.random.normal(k1, (num_tokens, num_kv_heads, head_dim), dtype=dtype)
    value = jax.random.normal(k2, (num_tokens, num_kv_heads, head_dim), dtype=dtype)
    key_cache = jax.random.normal(k3, (num_blocks, block_size, num_kv_heads, head_dim), dtype=dtype)
    value_cache = jax.random.normal(k4, (num_blocks, block_size, num_kv_heads, head_dim), dtype=dtype)

    total_slots = num_blocks * block_size
    perm = jax.random.permutation(k5, total_slots)[:num_tokens]
    valid_mask = jnp.arange(num_tokens) < (num_tokens - 64)
    slot_mapping = jnp.where(valid_mask, perm, -1).astype(jnp.int32)

    return key, value, key_cache, value_cache, slot_mapping


def workload(key, value, key_cache, value_cache, slot_mapping):
    num_blocks = CONFIG['num_blocks']
    block_size = CONFIG['block_size']
    num_kv_heads = CONFIG['num_kv_heads']
    head_dim = CONFIG['head_dim']
    total_slots = num_blocks * block_size

    flat_key_cache = key_cache.reshape(total_slots, num_kv_heads, head_dim)
    flat_value_cache = value_cache.reshape(total_slots, num_kv_heads, head_dim)

    safe_slots = jnp.where(slot_mapping >= 0, slot_mapping, total_slots)

    updated_key_cache = flat_key_cache.at[safe_slots].set(key, mode='drop')
    updated_value_cache = flat_value_cache.at[safe_slots].set(value, mode='drop')

    return (
        updated_key_cache.reshape(num_blocks, block_size, num_kv_heads, head_dim),
        updated_value_cache.reshape(num_blocks, block_size, num_kv_heads, head_dim),
    )
