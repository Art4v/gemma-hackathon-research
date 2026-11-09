"""[m1 M4_split_k variant of authored/softcapped_logits/gemma_softcapped_logits_logsoftmax_2] Tied-embedding LM head -> tanh soft-cap -> log-softmax with the contraction (hidden) dim grown 2304 -> 12288 and the vocab shrunk to 32768."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'gemma_softcapped_logits_logsoftmax_m1_splitk',
    'num_tokens': 1024,
    'hidden_dim': 12288,
    'vocab_size': 32768,
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
