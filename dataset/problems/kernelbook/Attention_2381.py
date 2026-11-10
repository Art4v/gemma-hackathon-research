"""[bf16 variant] Dot-product attention: softmax(query @ keys^T, axis=-1) @ keys, returning (contexts, attn_weights)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Attention_2381',
    'batch': 16,
    'query_len': 2048,
    'key_len': 2048,
    'input_size': 512,
}


def create_inputs(dtype=jnp.bfloat16):
    kq, kk = jax.random.split(jax.random.key(0))
    query = jax.random.uniform(
        kq, (CONFIG['batch'], CONFIG['query_len'], CONFIG['input_size']), dtype)
    keys = jax.random.uniform(
        kk, (CONFIG['batch'], CONFIG['key_len'], CONFIG['input_size']), dtype)
    return query, keys


def workload(query, keys):
    align = jnp.einsum('bqd,bkd->bqk', query, keys)
    attn_weights = jax.nn.softmax(align, axis=-1)
    contexts = jnp.einsum('bqk,bkd->bqd', attn_weights, keys)
    return contexts, attn_weights
