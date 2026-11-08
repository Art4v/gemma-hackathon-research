"""Fused LM-head linear projection -> final-logit tanh softcap -> per-token cross-entropy loss (chunked over tokens)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'fused_linear_softcap_cross_entropy',
    'num_tokens': 8192,
    'hidden_dim': 1152,
    'vocab_size': 256000,
    'chunk_size': 512,
    'softcap': 30.0,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k_x, k_w, k_l = jax.random.split(key, 3)
    n = CONFIG['num_tokens']
    d = CONFIG['hidden_dim']
    v = CONFIG['vocab_size']
    hidden = jax.random.normal(k_x, (n, d), dtype)
    weight = jax.random.normal(k_w, (v, d), dtype) * 0.02
    labels = jax.random.randint(k_l, (n,), 0, v, dtype=jnp.int32)
    return hidden, weight, labels


def workload(hidden, weight, labels):
    n, d = hidden.shape
    chunk = CONFIG['chunk_size']
    cap = CONFIG['softcap']
    num_chunks = n // chunk

    xc = hidden.reshape(num_chunks, chunk, d)
    lc = labels.reshape(num_chunks, chunk)

    def per_chunk(args):
        xb, lb = args
        logits = lax.dot_general(
            xb, weight,
            dimension_numbers=(((1,), (1,)), ((), ())),
            preferred_element_type=jnp.float32,
        )  # [chunk, V] float32
        logits = cap * jnp.tanh(logits / cap)
        m = jnp.max(logits, axis=-1, keepdims=True)
        lse = jnp.log(jnp.sum(jnp.exp(logits - m), axis=-1)) + m[:, 0]
        tgt = jnp.take_along_axis(logits, lb[:, None], axis=-1)[:, 0]
        return lse - tgt

    losses = lax.map(per_chunk, (xc, lc))
    return losses.reshape(n)
