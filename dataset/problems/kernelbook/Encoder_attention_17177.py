"""Encoder_attention: Linear(n_h->1) -> softmax over dim 1 -> broadcast multiply with x -> sum over dim 1."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Encoder_attention_17177',
    'batch': 8,
    'seq': 128,
    'n_a': 256,   # must equal n_h: torch broadcasts weights (B,S,1,A) against x (B,S,A,H)
    'n_h': 256,
}


def create_inputs(dtype=jnp.float32):
    kx, kw, kb = jax.random.split(jax.random.key(0), 3)
    x = jax.random.uniform(kx, (CONFIG['batch'], CONFIG['seq'], CONFIG['n_a'], CONFIG['n_h']), dtype)
    w = jax.random.normal(kw, (CONFIG['n_h'], 1), dtype) * 0.02
    b = jax.random.normal(kb, (1,), dtype) * 0.02
    return x, w, b


def workload(x, w, b):
    logits = (jnp.matmul(x, w, precision='highest') + b)[..., 0]  # (B, S, A)
    weights = jax.nn.softmax(logits, axis=1)
    x2 = jnp.sum(x * weights[:, :, None, :], axis=1)  # (B, A, H)
    return x2, weights
