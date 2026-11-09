"""[bf16 variant] QueryEncoding: x + Embedding(2, d_model)[idx], where idx = 0 at N-position 0, else 1."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'QueryEncoding_4551',
    'batch': 8,
    'n_queries': 16,
    'seq_len': 1024,
    'd_model': 512,
    'num_embeddings': 2,
}


def create_inputs(dtype=jnp.bfloat16):
    k_x, k_pe = jax.random.split(jax.random.key(0))
    x = jax.random.uniform(
        k_x,
        (CONFIG['batch'], CONFIG['n_queries'], CONFIG['seq_len'], CONFIG['d_model']),
        dtype,
    )
    pe = jax.random.normal(k_pe, (CONFIG['num_embeddings'], CONFIG['d_model']), dtype) * 0.02
    return x, pe


def workload(x, pe):
    n = x.shape[1]
    # idx depends only on the N axis: row 0 for the first query slot, row 1 for the rest.
    idx = jnp.where(jnp.arange(n) == 0, 0, 1)
    emb = jnp.take(pe, idx, axis=0)  # (N, K)
    return x + emb[None, :, None, :].astype(x.dtype)
