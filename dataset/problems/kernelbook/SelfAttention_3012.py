"""SelfAttention_3012: q/k/v linear projections (no bias) -> softmax(q k^T / sqrt(dim_k)) -> @ v."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'SelfAttention_3012',
    'batch': 16,
    'seq_len': 2048,
    'dim_in': 1024,
    'dim_k': 1024,
    'dim_v': 1024,
}


def create_inputs(dtype=jnp.float32):
    kx, kq, kk, kv = jax.random.split(jax.random.key(0), 4)
    b, n = CONFIG['batch'], CONFIG['seq_len']
    d_in, d_k, d_v = CONFIG['dim_in'], CONFIG['dim_k'], CONFIG['dim_v']
    x = jax.random.uniform(kx, (b, n, d_in), dtype)
    w_q = jax.random.normal(kq, (d_in, d_k), dtype) * 0.02
    w_k = jax.random.normal(kk, (d_in, d_k), dtype) * 0.02
    w_v = jax.random.normal(kv, (d_in, d_v), dtype) * 0.02
    return x, w_q, w_k, w_v


def workload(x, w_q, w_k, w_v):
    q = x @ w_q
    k = x @ w_k
    v = x @ w_v
    norm_fact = 1.0 / jnp.sqrt(jnp.asarray(w_k.shape[1], x.dtype))
    dist = jnp.einsum('bqd,bkd->bqk', q, k) * norm_fact
    dist = jax.nn.softmax(dist, axis=-1)
    return jnp.einsum('bqk,bkd->bqd', dist, v)
