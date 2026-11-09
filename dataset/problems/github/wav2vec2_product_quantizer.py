"""[bf16 variant] wav2vec2 product quantizer (eval): linear -> per-group argmax one-hot -> codevector lookup + masked perplexity."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'wav2vec2_product_quantizer',
    'batch': 16,
    'seq_len': 4096,
    'conv_dim': 512,
    'num_groups': 2,
    'num_vars': 320,
    'codevector_dim': 256,
    'mask_threshold': 0.35,
}


def create_inputs(dtype=jnp.bfloat16):
    B, T, C = CONFIG['batch'], CONFIG['seq_len'], CONFIG['conv_dim']
    G, V, D = CONFIG['num_groups'], CONFIG['num_vars'], CONFIG['codevector_dim']
    k = jax.random.split(jax.random.key(0), 5)
    hidden_states = jax.random.normal(k[0], (B, T, C), dtype)
    mask_time_indices = jax.random.uniform(k[1], (B, T)) > CONFIG['mask_threshold']
    w = jax.random.normal(k[2], (C, G * V), dtype) * 0.02
    b = jax.random.normal(k[3], (G * V,), dtype) * 0.02
    codevectors = jax.random.normal(k[4], (G * V, D // G), dtype) * 0.02
    return hidden_states, mask_time_indices, w, b, codevectors


def workload(hidden_states, mask_time_indices, w, b, codevectors):
    B, T, _ = hidden_states.shape
    G, V = CONFIG['num_groups'], CONFIG['num_vars']
    N = B * T
    logits = jnp.dot(hidden_states.reshape(N, -1), w, precision=lax.Precision.HIGHEST) + b
    idx = jnp.argmax(logits.reshape(N, G, V), axis=-1)  # (N, G)

    # Perplexity of the masked marginal code distribution.
    probs = jax.nn.one_hot(idx, V, dtype=hidden_states.dtype)  # (N, G, V)
    mask = mask_time_indices.reshape(N)
    probs = jnp.where(mask[:, None, None], probs, jnp.zeros_like(probs))
    marginal = probs.sum(axis=0) / mask.sum().astype(probs.dtype)
    safe = jnp.where(marginal > 0, marginal, jnp.ones_like(marginal))
    plogp = jnp.where(marginal > 0, marginal * jnp.log(safe), jnp.zeros_like(marginal))
    perplexity = jnp.exp(-jnp.sum(plogp, axis=-1)).sum()

    # One-hot @ codevectors selects exactly one row per group: exact gather.
    flat_idx = idx + jnp.arange(G)[None, :] * V
    out = jnp.take(codevectors, flat_idx, axis=0)  # (N, G, D/G)
    return out.reshape(B, T, -1), perplexity
