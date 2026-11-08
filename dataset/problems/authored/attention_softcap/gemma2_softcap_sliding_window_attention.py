"""Gemma 2 attention: query pre-scale -> QK^T -> tanh logit soft-cap -> causal sliding-window mask -> f32 softmax -> PV."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'gemma2_softcap_sliding_window_attention',
    'batch': 1,
    'num_heads': 16,
    'seq_len': 4096,
    'head_dim': 256,
    'sliding_window': 1024,
    'attn_logit_softcap': 50.0,
    'query_pre_attn_scalar': 256.0,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    kq, kk, kv = jax.random.split(key, 3)
    shape = (CONFIG['batch'], CONFIG['num_heads'], CONFIG['seq_len'], CONFIG['head_dim'])
    q = jax.random.normal(kq, shape, dtype)
    k = jax.random.normal(kk, shape, dtype)
    v = jax.random.normal(kv, shape, dtype)
    return q, k, v


def workload(q, k, v):
    seq_len = q.shape[2]
    scale = CONFIG['query_pre_attn_scalar'] ** -0.5
    cap = CONFIG['attn_logit_softcap']
    window = CONFIG['sliding_window']

    qs = (q.astype(jnp.float32) * scale).astype(q.dtype)
    logits = jnp.einsum('bhqd,bhkd->bhqk', qs, k,
                        preferred_element_type=jnp.float32)
    logits = cap * jnp.tanh(logits / cap)

    qi = jnp.arange(seq_len)[:, None]
    kj = jnp.arange(seq_len)[None, :]
    mask = (kj <= qi) & (kj > qi - window)
    logits = jnp.where(mask[None, None], logits, jnp.finfo(jnp.float32).min)

    probs = jax.nn.softmax(logits, axis=-1)
    out = jnp.einsum('bhqk,bhkd->bhqd', probs, v.astype(jnp.float32),
                     preferred_element_type=jnp.float32)
    return out.astype(q.dtype)
