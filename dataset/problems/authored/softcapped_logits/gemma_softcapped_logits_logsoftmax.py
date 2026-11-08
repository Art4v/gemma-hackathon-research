"""Gemma LM head: logits = h @ E^T -> tanh soft-cap -> log_softmax over vocab (bf16 in, f32 out)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'gemma_softcapped_logits_logsoftmax',
    'num_tokens': 512,
    'hidden_dim': 1152,
    'vocab_size': 262144,
    'final_logit_softcap': 30.0,
    'hidden_scale': 30.0,
    'weight_std': 0.02,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    k_h, k_w = jax.random.split(key)
    T = CONFIG['num_tokens']
    D = CONFIG['hidden_dim']
    V = CONFIG['vocab_size']
    hidden = (jax.random.normal(k_h, (T, D), jnp.float32) * CONFIG['hidden_scale']).astype(dtype)
    embedding = jax.random.normal(k_w, (V, D), dtype) * CONFIG['weight_std']
    return hidden, embedding


def workload(hidden, embedding):
    cap = CONFIG['final_logit_softcap']
    # [T, D] x [V, D]^T -> [T, V], accumulate in float32.
    logits = lax.dot_general(
        hidden,
        embedding,
        dimension_numbers=(((1,), (1,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    logits = cap * jnp.tanh(logits / cap)
    m = jnp.max(logits, axis=-1, keepdims=True)
    shifted = logits - m
    lse = jnp.log(jnp.sum(jnp.exp(shifted), axis=-1, keepdims=True))
    return (shifted - lse).astype(jnp.float32)
