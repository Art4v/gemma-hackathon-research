"""Paged KV-cache write: scatter new K/V rows into a block-paged bf16 cache by slot mapping."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'paged_kv_cache_scatter_update',
    'num_blocks': 4096,
    'block_size': 16,
    'num_kv_heads': 8,
    'head_dim': 128,
    'num_tokens': 8192,
}


def create_inputs(dtype=jnp.float32):
    num_blocks = CONFIG['num_blocks']
    block_size = CONFIG['block_size']
    num_kv_heads = CONFIG['num_kv_heads']
    head_dim = CONFIG['head_dim']
    num_tokens = CONFIG['num_tokens']
    num_slots = num_blocks * block_size

    k_cache, k_newk, k_newv, k_slot = jax.random.split(jax.random.key(0), 4)

    # Fused K/V paged cache in vLLM-TPU layout: [num_blocks, block_size, 2*num_kv_heads, head_dim]
    kv_cache = jax.random.normal(
        k_cache, (num_blocks, block_size, 2 * num_kv_heads, head_dim), dtype)
    # New K and V for this step's tokens: [num_tokens, num_kv_heads, head_dim]
    new_k = jax.random.normal(k_newk, (num_tokens, num_kv_heads, head_dim), dtype)
    new_v = jax.random.normal(k_newv, (num_tokens, num_kv_heads, head_dim), dtype)
    # Flat slot index per token (block_id * block_size + offset); unique so the scatter is deterministic.
    slot_mapping = jax.random.permutation(k_slot, num_slots)[:num_tokens].astype(jnp.int32)

    return (kv_cache, new_k, new_v, slot_mapping)


def workload(kv_cache, new_k, new_v, slot_mapping):
    num_blocks, block_size, kv_heads2, head_dim = kv_cache.shape
    num_slots = num_blocks * block_size

    # Fuse K and V along the head axis to match the cache layout, cast to cache dtype.
    new_kv = jnp.concatenate([new_k, new_v], axis=1).astype(kv_cache.dtype)

    # View the paged cache as a flat slot table and scatter the new rows.
    flat_cache = kv_cache.reshape(num_slots, kv_heads2, head_dim)
    flat_cache = flat_cache.at[slot_mapping].set(new_kv, unique_indices=True)

    return flat_cache.reshape(num_blocks, block_size, kv_heads2, head_dim)
