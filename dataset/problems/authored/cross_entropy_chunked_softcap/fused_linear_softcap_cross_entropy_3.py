"""Fused LM-head linear projection -> final-logit tanh softcap -> per-token cross-entropy (Gemma 2/3 style)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'fused_linear_softcap_cross_entropy',
    'num_tokens': 8192,
    'hidden_dim': 1152,
    'vocab_size': 262144,
    'softcap': 30.0,
    'ignore_index': -100,
    'ignore_fraction_denominator': 16,
    'token_chunk': 256,
}


def create_inputs(dtype=jnp.float32):
    T = CONFIG['num_tokens']
    D = CONFIG['hidden_dim']
    V = CONFIG['vocab_size']
    key = jax.random.key(0)
    k_h, k_w, k_lbl, k_mask = jax.random.split(key, 4)
    hidden = jax.random.normal(k_h, (T, D), dtype)
    weight = jax.random.normal(k_w, (D, V), dtype) * 0.02
    labels = jax.random.randint(k_lbl, (T,), 0, V, dtype=jnp.int32)
    drop = jax.random.randint(k_mask, (T,), 0, CONFIG['ignore_fraction_denominator'], dtype=jnp.int32) == 0
    labels = jnp.where(drop, jnp.int32(CONFIG['ignore_index']), labels)
    return hidden, labels, weight


def workload(hidden, labels, weight):
    T, D = hidden.shape
    cap = CONFIG['softcap']
    ignore_index = CONFIG['ignore_index']
    chunk = min(CONFIG['token_chunk'], T)
    n_chunks = T // chunk

    h_chunks = hidden.reshape(n_chunks, chunk, D)
    l_chunks = labels.reshape(n_chunks, chunk)

    def per_chunk(args):
        h, lbl = args
        logits = jnp.dot(h, weight, preferred_element_type=jnp.float32)  # [C, V] f32
        capped = cap * jnp.tanh(logits / cap)
        m = jnp.max(capped, axis=-1, keepdims=True)
        lse = jnp.log(jnp.sum(jnp.exp(capped - m), axis=-1)) + m[:, 0]
        valid = lbl != ignore_index
        safe_lbl = jnp.where(valid, lbl, 0)
        target = jnp.take_along_axis(capped, safe_lbl[:, None], axis=-1)[:, 0]
        return jnp.where(valid, lse - target, 0.0).astype(jnp.float32)

    losses = lax.map(per_chunk, (h_chunks, l_chunks))
    return losses.reshape(T)
