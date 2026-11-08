"""Tied-embedding LM head matmul -> tanh logit soft-capping (Gemma 2) -> log-softmax over vocab, f32 out."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'gemma_softcapped_logits_logsoftmax',
    'num_tokens': 1024,
    'hidden_dim': 2304,
    'vocab_size': 131072,
    'final_logit_softcap': 30.0,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    k_h, k_w = jax.random.split(key)
    T = CONFIG['num_tokens']
    D = CONFIG['hidden_dim']
    V = CONFIG['vocab_size']
    hidden = jax.random.normal(k_h, (T, D), dtype)
    embedding = jax.random.normal(k_w, (V, D), dtype) * CONFIG['weight_scale']
    return hidden, embedding


def workload(hidden, embedding):
    cap = CONFIG['final_logit_softcap']
    # Vocab projection with tied embeddings, f32 accumulation.
    logits = jnp.einsum('td,vd->tv', hidden, embedding,
                        preferred_element_type=jnp.float32)
    # Gemma-style tanh soft-capping.
    logits = cap * jnp.tanh(logits / cap)
    # Numerically stable log-softmax over the vocab axis, in f32.
    m = jnp.max(logits, axis=-1, keepdims=True)
    shifted = logits - m
    lse = jnp.log(jnp.sum(jnp.exp(shifted), axis=-1, keepdims=True))
    return (shifted - lse).astype(jnp.float32)
