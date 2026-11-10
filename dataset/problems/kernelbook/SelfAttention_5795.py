"""[bf16 variant] SelfAttention_5795: affine query and key projections, batched Q @ K^T, divided by sqrt(data_dim)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'SelfAttention_5795',
    'batch': 8,
    'seq_len': 1024,
    'data_dim': 2048,
    'dim_q': 1024,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.bfloat16):
    keys = jax.random.split(jax.random.key(0), 5)
    batch = CONFIG['batch']
    seq_len = CONFIG['seq_len']
    data_dim = CONFIG['data_dim']
    dim_q = CONFIG['dim_q']
    scale = CONFIG['weight_scale']
    x = jax.random.uniform(keys[0], (batch, seq_len, data_dim), dtype)
    w_q = jax.random.normal(keys[1], (data_dim, dim_q), dtype) * scale
    b_q = jax.random.normal(keys[2], (dim_q,), dtype) * scale
    w_k = jax.random.normal(keys[3], (data_dim, dim_q), dtype) * scale
    b_k = jax.random.normal(keys[4], (dim_q,), dtype) * scale
    return x, w_q, b_q, w_k, b_k


def workload(x, w_q, b_q, w_k, b_k):
    queries = x @ w_q + b_q
    keys = x @ w_k + b_k
    dot = queries @ jnp.swapaxes(keys, 1, 2)
    return dot / jnp.sqrt(jnp.asarray(x.shape[-1], x.dtype))
