"""Gemma 2 style causal MHA: QK^T * scale -> tanh soft-cap -> causal mask -> f32 softmax -> PV."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'gemma2_softcap_causal_attention',
    'batch': 1,
    'num_heads': 16,
    'seq_len': 4096,
    'head_dim': 256,
    'query_pre_attn_scalar': 256,
    'attn_logit_softcap': 50.0,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    kq, kk, kv = jax.random.split(key, 3)
    shape = (CONFIG['batch'], CONFIG['num_heads'], CONFIG['seq_len'], CONFIG['head_dim'])
    q = jax.random.normal(kq, shape, dtype)
    k = jax.random.normal(kk, shape, dtype)
    v = jax.random.normal(kv, shape, dtype)
    return q, k, v


def workload(q, k, v):
    scale = CONFIG['query_pre_attn_scalar'] ** -0.5
    cap = CONFIG['attn_logit_softcap']
    seq_q = q.shape[2]
    seq_k = k.shape[2]

    logits = jnp.einsum('bhqd,bhkd->bhqk', q, k,
                        preferred_element_type=jnp.float32)
    logits = logits * scale
    logits = cap * jnp.tanh(logits / cap)

    q_idx = jnp.arange(seq_q)[:, None]
    k_idx = jnp.arange(seq_k)[None, :]
    causal = k_idx <= q_idx
    logits = jnp.where(causal[None, None, :, :], logits, jnp.finfo(jnp.float32).min)

    probs = jax.nn.softmax(logits, axis=-1)
    out = jnp.einsum('bhqk,bhkd->bhqd', probs, v.astype(jnp.float32),
                     preferred_element_type=jnp.float32)
    return out.astype(q.dtype)
