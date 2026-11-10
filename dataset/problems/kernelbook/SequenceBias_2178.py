"""[bf16 variant] SequenceBias: tile a learned bias to (1, N, E) and concatenate it after (L, N, E) along the sequence axis."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'SequenceBias_2178',
    'seq_len': 1024,
    'batch': 128,
    'embed_dim': 256,
    'bias_scale': 0.02,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k_x, k_bias = jax.random.split(key)
    x = jax.random.uniform(
        k_x,
        (CONFIG['seq_len'], CONFIG['batch'], CONFIG['embed_dim']),
        dtype,
    )
    bias = jax.random.normal(k_bias, (CONFIG['embed_dim'],), dtype) * CONFIG['bias_scale']
    return x, bias


def workload(x, bias):
    _, bsz, embed_dim = x.shape
    tiled = jnp.broadcast_to(bias, (1, bsz, embed_dim))
    return jnp.concatenate([x, tiled], axis=0)
