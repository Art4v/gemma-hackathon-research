"""Multi-head attention (key used as value, no scaling) -> concat with query -> Linear -> tanh."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'MultiHeadAttention_7344',
    'batch': 16,
    'query_len': 2048,
    'key_len': 2048,
    'in_features': 512,
    'n_head': 4,
    'dim': 128,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    f, hd = c['in_features'], c['n_head'] * c['dim']
    k = jax.random.split(jax.random.key(0), 5)
    query = jax.random.uniform(k[0], (c['batch'], c['query_len'], f), dtype)
    key = jax.random.uniform(k[1], (c['batch'], c['key_len'], f), dtype)
    wq = jax.random.normal(k[2], (f, hd), dtype) * 0.02
    bq = jnp.zeros((hd,), dtype)
    wk = jax.random.normal(k[3], (f, hd), dtype) * 0.02
    bk = jnp.zeros((hd,), dtype)
    wout = jax.random.normal(k[4], (2 * f, f), dtype) * 0.02
    bout = jnp.zeros((f,), dtype)
    return query, key, wq, bq, wk, bk, wout, bout


def workload(query, key, wq, bq, wk, bk, wout, bout):
    b, lq, _ = query.shape
    lk = key.shape[1]
    h = CONFIG['n_head']
    d = wq.shape[1] // h
    f = wout.shape[1]
    q = (query @ wq + bq).reshape(b, lq, h, d)
    kp = (key @ wk + bk).reshape(b, lk, h, d)
    scores = jnp.einsum('bqhd,bkhd->bhqk', q, kp)
    attn = jax.nn.softmax(scores, axis=-1)
    # The source uses the projected key as the value too.
    context = jnp.einsum('bhqk,bkhd->bqhd', attn, kp).reshape(b, lq, h * d)
    combined = jnp.concatenate([context, query], axis=2)
    # Same flat reshape as torch's view(-1, 2F); it only lines up row by row when h*d == F.
    out = jnp.tanh(combined.reshape(-1, 2 * f) @ wout + bout)
    return out.reshape(b, -1, f)
