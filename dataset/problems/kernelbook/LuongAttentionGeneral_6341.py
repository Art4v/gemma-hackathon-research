"""Luong general attention: Linear(values) @ broadcast query^T, softmax over dim 1, weighted sum of values."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'LuongAttentionGeneral_6341',
    'n_query': 8,   # N: leading dim of query, survives squeeze(0) since != 1
    'n_batch': 8,   # M: dim 1 of query == dim 0 of values (broadcast + softmax axis)
    'seq': 128,     # S: dim 1 of values
    'rows': 128,    # C: dim 2 of values
    'hidden': 128,  # H: Linear size; query is (N, M, H, H) so matmul shapes line up
}


def create_inputs(dtype=jnp.float32):
    n, m, s = CONFIG['n_query'], CONFIG['n_batch'], CONFIG['seq']
    c, h = CONFIG['rows'], CONFIG['hidden']
    kq, kv, kw, kb = jax.random.split(jax.random.key(0), 4)
    query = jax.random.uniform(kq, (n, m, h, h), dtype)
    values = jax.random.uniform(kv, (m, s, c, h), dtype)
    w = jax.random.normal(kw, (h, h), dtype) * 0.02
    b = jax.random.normal(kb, (h,), dtype) * 0.02
    return query, values, w, b


def workload(query, values, w, b):
    wv = values @ w + b  # (M, S, C, H)
    # torch.matmul broadcast of (M,S,C,H) with (N,M,1,H,H) -> (N,M,S,C,H)
    score = jnp.einsum('jkcm,ijmh->ijkch', wv, query)
    attention_weights = jax.nn.softmax(score, axis=1)
    context_vector = jnp.sum(attention_weights * values[None], axis=1)
    return context_vector, attention_weights
