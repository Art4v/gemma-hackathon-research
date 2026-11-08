"""Fused linear projection, logit soft-capping, and cross-entropy loss from Gemma-2 9B."""

import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'gemma2_9b_linear_softcap_cross_entropy',
    'num_tokens': 8192,
    'hidden_size': 3584,
    'vocab_size': 131072,
    'vocab_chunk_size': 8192,
    'softcap': 30.0,
    'z_loss_weight': 1e-4,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k1, k2, k3 = jax.random.split(key, 3)
    n = CONFIG['num_tokens']
    d = CONFIG['hidden_size']
    v = CONFIG['vocab_size']

    hidden_states = jax.random.normal(k1, (n, d), dtype=dtype)
    labels = jax.random.randint(k2, (n,), 0, v, dtype=jnp.int32)
    lm_head_weight = jax.random.normal(k3, (v, d), dtype=dtype) * 0.02
    return hidden_states, labels, lm_head_weight


def workload(hidden_states, labels, lm_head_weight):
    n = hidden_states.shape[0]
    d = hidden_states.shape[1]
    v = lm_head_weight.shape[0]
    v_chunk = CONFIG['vocab_chunk_size']
    softcap = CONFIG['softcap']
    z_loss_weight = CONFIG['z_loss_weight']

    num_chunks = v // v_chunk
    weight_chunks = lm_head_weight.reshape(num_chunks, v_chunk, d)

    init_max = jnp.full((n,), -jnp.inf, dtype=jnp.float32)
    init_sum = jnp.zeros((n,), dtype=jnp.float32)
    init_target = jnp.zeros((n,), dtype=jnp.float32)

    def body_fn(carry, idx):
        m_prev, s_prev, target_logit = carry
        w_c = weight_chunks[idx]

        raw_logits = jnp.dot(
            hidden_states,
            w_c.T,
            preferred_element_type=jnp.float32,
        )
        logits = softcap * jnp.tanh(raw_logits / softcap)

        m_curr = jnp.max(logits, axis=-1)
        m_new = jnp.maximum(m_prev, m_curr)
        s_new = s_prev * jnp.exp(m_prev - m_new) + jnp.sum(
            jnp.exp(logits - m_new[:, None]), axis=-1
        )

        chunk_start = idx * v_chunk
        in_chunk = (labels >= chunk_start) & (labels < chunk_start + v_chunk)
        local_labels = jnp.where(in_chunk, labels - chunk_start, 0)
        picked = jnp.take_along_axis(
            logits, local_labels[:, None], axis=-1
        ).squeeze(-1)
        target_logit = target_logit + jnp.where(in_chunk, picked, 0.0)

        return (m_new, s_new, target_logit), None

    (m_final, s_final, target_logits), _ = lax.scan(
        body_fn,
        (init_max, init_sum, init_target),
        jnp.arange(num_chunks),
    )

    lse = m_final + jnp.log(s_final)
    ce_loss = lse - target_logits
    z_loss = z_loss_weight * jnp.square(lse)
    return ce_loss + z_loss
