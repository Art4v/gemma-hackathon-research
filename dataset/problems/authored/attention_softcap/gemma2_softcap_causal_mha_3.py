"""Gemma 2 causal multi-head attention with tanh logit soft-capping."""

import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'gemma2_softcap_causal_mha',
    'batch': 4,
    'num_heads': 16,
    'seq_len': 2048,
    'head_dim': 256,
    'softcap': 50.0,
    'sm_scale': 0.0625,
    'mask_value': -1e30,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k_q, k_k, k_v = jax.random.split(key, 3)
    shape = (
        CONFIG['batch'],
        CONFIG['num_heads'],
        CONFIG['seq_len'],
        CONFIG['head_dim'],
    )
    q = jax.random.normal(k_q, shape, dtype=dtype)
    k = jax.random.normal(k_k, shape, dtype=dtype)
    v = jax.random.normal(k_v, shape, dtype=dtype)
    return q, k, v


def workload(q, k, v):
    softcap = CONFIG['softcap']
    sm_scale = CONFIG['sm_scale']
    mask_value = CONFIG['mask_value']
    seq_len = q.shape[2]

    logits = jnp.einsum(
        'bhqd,bhkd->bhqk',
        q,
        k,
        preferred_element_type=jnp.float32,
    )
    logits = logits * sm_scale
    logits = softcap * jnp.tanh(logits / softcap)

    q_idx = jnp.arange(seq_len)[:, None]
    k_idx = jnp.arange(seq_len)[None, :]
    causal_mask = q_idx >= k_idx
    logits = jnp.where(causal_mask[None, None, :, :], logits, mask_value)

    max_logits = jnp.max(logits, axis=-1, keepdims=True)
    exp_logits = jnp.exp(logits - max_logits)
    probs = exp_logits / jnp.sum(exp_logits, axis=-1, keepdims=True)

    out = jnp.einsum(
        'bhqk,bhkd->bhqd',
        probs.astype(v.dtype),
        v,
        preferred_element_type=jnp.float32,
    )
    return out.astype(q.dtype)
