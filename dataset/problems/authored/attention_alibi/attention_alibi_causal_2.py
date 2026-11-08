"""Multi-head causal attention with ALiBi linear positional bias."""

import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'attention_alibi_causal',
    'batch_size': 4,
    'seq_len': 2048,
    'num_heads': 32,
    'head_dim': 128,
    'alibi_base': 8.0,
    'mask_value': -1e30,
}


def create_inputs(dtype=jnp.bfloat16):
    b = CONFIG['batch_size']
    s = CONFIG['seq_len']
    h = CONFIG['num_heads']
    d = CONFIG['head_dim']
    base = CONFIG['alibi_base']

    key = jax.random.key(0)
    k_q, k_k, k_v = jax.random.split(key, 3)

    q = jax.random.normal(k_q, (b, s, h, d), dtype=dtype)
    k = jax.random.normal(k_k, (b, s, h, d), dtype=dtype)
    v = jax.random.normal(k_v, (b, s, h, d), dtype=dtype)

    head_indices = jnp.arange(1, h + 1, dtype=jnp.float32)
    slopes = jnp.power(2.0, -base * head_indices / h)

    return q, k, v, slopes


def workload(q, k, v, slopes):
    b, s, h, d = q.shape
    scale = 1.0 / jnp.sqrt(jnp.float32(d))

    q_t = jnp.transpose(q, (0, 2, 1, 3))
    k_t = jnp.transpose(k, (0, 2, 1, 3))
    v_t = jnp.transpose(v, (0, 2, 1, 3))

    logits = jnp.matmul(q_t, jnp.swapaxes(k_t, -2, -1), preferred_element_type=jnp.float32) * scale

    pos = jnp.arange(s, dtype=jnp.float32)
    rel_pos = pos[None, :] - pos[:, None]
    alibi_bias = slopes[None, :, None, None] * rel_pos[None, None, :, :]
    logits = logits + alibi_bias

    causal_mask = rel_pos <= 0
    logits = jnp.where(causal_mask[None, None, :, :], logits, CONFIG['mask_value'])

    probs = jax.nn.softmax(logits, axis=-1).astype(q.dtype)
    out = jnp.matmul(probs, v_t, preferred_element_type=jnp.float32).astype(q.dtype)
    return jnp.transpose(out, (0, 2, 1, 3))
