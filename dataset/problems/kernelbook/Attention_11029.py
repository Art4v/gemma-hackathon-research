"""Attention_11029: tile+reshape query, per-row cosine similarity with per-element eps, affine layer, softmax over the 1-D result."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Attention_11029',
    'n_k': 2048,
    'pu_rows': 128,
    'pu_cols': 128,
    'mp_cols': 16384,
    'eps': 1e-05,
    'weight_scale': 0.02,
    'sum_axis': 1,
    'softmax_axis': 0,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    k_pu, k_mp, k_w, k_b = jax.random.split(key, 4)
    n_k = CONFIG['n_k']
    pu = jax.random.uniform(k_pu, (CONFIG['pu_rows'], CONFIG['pu_cols']), dtype)
    mp = jax.random.uniform(k_mp, (n_k, CONFIG['mp_cols']), dtype)
    w = jax.random.normal(k_w, (n_k, n_k), dtype) * CONFIG['weight_scale']
    b = jax.random.normal(k_b, (n_k,), dtype) * CONFIG['weight_scale']
    return pu, mp, w, b


def workload(pu, mp, w, b):
    eps = CONFIG['eps']
    axis = CONFIG['sum_axis']
    m = mp.shape[0]
    expanded_pu = jnp.reshape(jnp.tile(pu, (1, m)), (m, -1))
    query_norm = jnp.sqrt(jnp.sum(expanded_pu ** 2 + eps, axis=axis))
    doc_norm = jnp.sqrt(jnp.sum(mp ** 2 + eps, axis=axis))
    prod = jnp.sum(expanded_pu * mp, axis=axis)
    cos_sim = prod / (query_norm * doc_norm)
    fc = cos_sim @ w + b
    return jax.nn.softmax(fc, axis=CONFIG['softmax_axis'])
