"""Gemma-2 style causal multi-head attention with tanh logit soft-capping."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'gemma2_softcap_causal_mha',
    'batch_size': 4,
    'num_heads': 16,
    'seq_len': 2048,
    'head_dim': 256,
    'query_pre_attn_scalar': 256.0,
    'attn_logit_softcapping': 50.0,
    'mask_value': -1e30,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k_q, k_k, k_v = jax.random.split(key, 3)
    shape = (
        CONFIG['batch_size'],
        CONFIG['num_heads'],
        CONFIG['seq_len'],
        CONFIG['head_dim'],
    )
    q = jax.random.normal(k_q, shape, dtype=dtype)
    k = jax.random.normal(k_k, shape, dtype=dtype)
    v = jax.random.normal(k_v, shape, dtype=dtype)
    return q, k, v


def workload(q, k, v):
    scale = CONFIG['query_pre_attn_scalar'] ** -0.5
    softcap = CONFIG['attn_logit_softcapping']
    seq_len = CONFIG['seq_len']

    logits = jnp.einsum(
        'bhsd,bhtd->bhst',
        q,
        k,
        preferred_element_type=jnp.float32,
    )
    logits = logits * scale
    logits = softcap * jnp.tanh(logits / softcap)

    row_idx = jnp.arange(seq_len)[:, None]
    col_idx = jnp.arange(seq_len)[None, :]
    causal_mask = row_idx >= col_idx
    logits = jnp.where(causal_mask[None, None, :, :], logits, CONFIG['mask_value'])

    weights = jax.nn.softmax(logits, axis=-1)
    out = jnp.einsum(
        'bhst,bhtd->bhsd',
        weights.astype(v.dtype),
        v,
        preferred_element_type=jnp.float32,
    )
    return out.astype(q.dtype)
