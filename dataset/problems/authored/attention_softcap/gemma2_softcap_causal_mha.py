"""Causal multi-head attention with Gemma-2 tanh logit soft-capping before softmax."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'gemma2_softcap_causal_mha',
    'batch_size': 4,
    'num_heads': 16,
    'seq_len': 2048,
    'head_dim': 256,
    'softcap': 50.0,
    'mask_value': -1e30,
}

def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k_q, k_k, k_v = jax.random.split(key, 3)
    shape = (CONFIG['batch_size'], CONFIG['num_heads'], CONFIG['seq_len'], CONFIG['head_dim'])
    q = jax.random.normal(k_q, shape, dtype=dtype)
    k = jax.random.normal(k_k, shape, dtype=dtype)
    v = jax.random.normal(k_v, shape, dtype=dtype)
    return (q, k, v)

def workload(q, k, v):
    head_dim = CONFIG['head_dim']
    seq_len = CONFIG['seq_len']
    softcap = CONFIG['softcap']
    mask_value = CONFIG['mask_value']

    scale = 1.0 / jnp.sqrt(jnp.float32(head_dim))
    logits = jnp.einsum('bhqd,bhkd->bhqk', q, k, preferred_element_type=jnp.float32)
    logits = logits * scale

    logits = softcap * jnp.tanh(logits / softcap)

    q_idx = jnp.arange(seq_len)[:, None]
    k_idx = jnp.arange(seq_len)[None, :]
    causal_mask = q_idx >= k_idx
    logits = jnp.where(causal_mask[None, None, :, :], logits, mask_value)

    max_logits = jnp.max(logits, axis=-1, keepdims=True)
    exp_logits = jnp.exp(logits - max_logits)
    probs = exp_logits / jnp.sum(exp_logits, axis=-1, keepdims=True)

    out = jnp.einsum('bhqk,bhkd->bhqd', probs, v.astype(jnp.float32), preferred_element_type=jnp.float32)
    return out.astype(q.dtype)
