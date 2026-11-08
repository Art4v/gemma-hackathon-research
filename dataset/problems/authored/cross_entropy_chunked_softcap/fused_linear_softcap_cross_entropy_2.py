"""Fused LM-head projection + tanh logit softcap + softmax cross-entropy (per-token losses), chunked over tokens."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'fused_linear_softcap_cross_entropy',
    'num_tokens': 4096,
    'hidden_dim': 2048,
    'vocab_size': 128256,
    'chunk_size': 512,
    'softcap': 30.0,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    k_h, k_w, k_l = jax.random.split(key, 3)
    T = CONFIG['num_tokens']
    D = CONFIG['hidden_dim']
    V = CONFIG['vocab_size']
    hidden = jax.random.normal(k_h, (T, D), dtype=dtype)
    weight = jax.random.normal(k_w, (D, V), dtype=dtype) * 0.02
    labels = jax.random.randint(k_l, (T,), 0, V, dtype=jnp.int32)
    return hidden, weight, labels


def workload(hidden, weight, labels):
    T, D = hidden.shape
    C = CONFIG['chunk_size']
    cap = CONFIG['softcap']
    n_chunks = T // C
    h_chunks = hidden.reshape(n_chunks, C, D)
    l_chunks = labels.reshape(n_chunks, C)

    def chunk_loss(args):
        h_c, l_c = args
        logits = jnp.dot(h_c, weight, preferred_element_type=jnp.float32)
        logits = cap * jnp.tanh(logits / cap)
        m = jnp.max(logits, axis=-1, keepdims=True)
        lse = jnp.log(jnp.sum(jnp.exp(logits - m), axis=-1)) + m[:, 0]
        target = jnp.take_along_axis(logits, l_c[:, None], axis=-1)[:, 0]
        return lse - target

    losses = lax.map(chunk_loss, (h_chunks, l_chunks))
    return losses.reshape(T)
