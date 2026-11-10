"""SelfAttention: shared per-head q/k/v Linear -> QK^T / sqrt(embed_size) -> softmax -> AV -> fc_out Linear."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'SelfAttention_10797',
    'batch': 8,
    'seq_len': 2048,
    'embed_size': 1024,
    'heads': 8,
}


def create_inputs(dtype=jnp.float32):
    n, l, e, h = CONFIG['batch'], CONFIG['seq_len'], CONFIG['embed_size'], CONFIG['heads']
    d = e // h
    ks = jax.random.split(jax.random.key(0), 8)
    queries = jax.random.uniform(ks[0], (n, l, e), dtype)
    keys = jax.random.uniform(ks[1], (n, l, e), dtype)
    values = jax.random.uniform(ks[2], (n, l, e), dtype)
    w_v = jax.random.normal(ks[3], (d, d), dtype) * 0.02
    w_k = jax.random.normal(ks[4], (d, d), dtype) * 0.02
    w_q = jax.random.normal(ks[5], (d, d), dtype) * 0.02
    w_out = jax.random.normal(ks[6], (e, e), dtype) * 0.02
    b_out = jax.random.normal(ks[7], (e,), dtype) * 0.02
    return queries, keys, values, w_v, w_k, w_q, w_out, b_out


def workload(queries, keys, values, w_v, w_k, w_q, w_out, b_out):
    d = w_q.shape[0]
    e = w_out.shape[1]
    h = w_out.shape[0] // d
    n, lq, lk, lv = queries.shape[0], queries.shape[1], keys.shape[1], values.shape[1]
    q = queries.reshape(n, lq, h, d) @ w_q
    k = keys.reshape(n, lk, h, d) @ w_k
    v = values.reshape(n, lv, h, d) @ w_v
    energy = jnp.einsum('nqhd,nkhd->nhqk', q, k)
    # Scale uses the full embed size, matching the PyTorch module.
    attn = jax.nn.softmax(energy / (e ** 0.5), axis=3)
    out = jnp.einsum('nhql,nlhd->nqhd', attn, v).reshape(n, lq, h * d)
    return out @ w_out + b_out
