"""Multi-head causal attention with ALiBi linear positional bias."""

import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'attention_alibi_causal',
    'batch_size': 4,
    'num_heads': 32,
    'seq_len': 4096,
    'head_dim': 128,
    'alibi_base': 8.0,
    'mask_value': -1e30,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k_q, k_k, k_v = jax.random.split(key, 3)
    b = CONFIG['batch_size']
    h = CONFIG['num_heads']
    s = CONFIG['seq_len']
    d = CONFIG['head_dim']

    q = jax.random.normal(k_q, (b, h, s, d), dtype=dtype)
    k = jax.random.normal(k_k, (b, h, s, d), dtype=dtype)
    v = jax.random.normal(k_v, (b, h, s, d), dtype=dtype)

    head_idx = jnp.arange(1, h + 1, dtype=jnp.float32)
    slopes = jnp.power(2.0, -CONFIG['alibi_base'] * head_idx / h)
    return q, k, v, slopes


def workload(q, k, v, slopes):
    _, _, s, d = q.shape
    scale = 1.0 / jnp.sqrt(jnp.float32(d))

    logits = jnp.einsum('bhsd,bhtd->bhst', q, k, preferred_element_type=jnp.float32) * scale

    pos = jnp.arange(s, dtype=jnp.int32)
    rel_pos = pos[None, :] - pos[:, None]
    alibi_bias = slopes[None, :, None, None] * rel_pos[None, None, :, :].astype(jnp.float32)
    logits = logits + alibi_bias

    causal_mask = rel_pos <= 0
    logits = jnp.where(causal_mask[None, None, :, :], logits, CONFIG['mask_value'])

    probs = jax.nn.softmax(logits, axis=-1).astype(v.dtype)
    out = jnp.einsum('bhst,bhtd->bhsd', probs, v, preferred_element_type=jnp.float32)
    return out.astype(q.dtype)
