"""Fused linear projection, logit softcapping, and per-token cross-entropy loss (Gemma-2 9B)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'gemma2_9b_linear_softcap_cross_entropy',
    'num_tokens': 8192,
    'hidden_dim': 3584,
    'vocab_size': 131072,
    'softcap': 30.0,
    'vocab_chunk_size': 16384,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k1, k2, k3 = jax.random.split(key, 3)
    num_tokens = CONFIG['num_tokens']
    hidden_dim = CONFIG['hidden_dim']
    vocab_size = CONFIG['vocab_size']

    hidden = jax.random.normal(k1, (num_tokens, hidden_dim), dtype=dtype)
    weight = jax.random.normal(k2, (hidden_dim, vocab_size), dtype=dtype) * 0.02
    labels = jax.random.randint(k3, (num_tokens,), 0, vocab_size, dtype=jnp.int32)
    return hidden, labels, weight


def workload(hidden, labels, weight):
    num_tokens = hidden.shape[0]
    vocab_size = weight.shape[1]
    chunk_size = CONFIG['vocab_chunk_size']
    softcap = CONFIG['softcap']
    num_chunks = vocab_size // chunk_size

    weight_chunked = weight.reshape(hidden.shape[1], num_chunks, chunk_size)
    weight_chunked = jnp.swapaxes(weight_chunked, 0, 1)

    init_max = jnp.full((num_tokens,), -jnp.inf, dtype=jnp.float32)
    init_sum = jnp.zeros((num_tokens,), dtype=jnp.float32)
    init_target = jnp.zeros((num_tokens,), dtype=jnp.float32)

    def step(carry, i):
        m_prev, s_prev, target_logit = carry
        w_c = weight_chunked[i]

        logits = jnp.dot(hidden, w_c, preferred_element_type=jnp.float32)
        logits = softcap * jnp.tanh(logits / softcap)

        m_curr = jnp.max(logits, axis=-1)
        m_new = jnp.maximum(m_prev, m_curr)
        s_new = s_prev * jnp.exp(m_prev - m_new) + jnp.sum(
            jnp.exp(logits - m_new[:, None]), axis=-1
        )

        chunk_start = i * chunk_size
        in_chunk = (labels >= chunk_start) & (labels < chunk_start + chunk_size)
        local_idx = jnp.where(in_chunk, labels - chunk_start, 0)
        gathered = jnp.take_along_axis(logits, local_idx[:, None], axis=-1).squeeze(-1)
        target_logit = jnp.where(in_chunk, gathered, target_logit)

        return (m_new, s_new, target_logit), None

    (m_final, s_final, target_final), _ = lax.scan(
        step, (init_max, init_sum, init_target), jnp.arange(num_chunks)
    )
    lse = m_final + jnp.log(s_final)
    loss = lse - target_final
    return loss
