"""MultAttention: three Linear encoders -> broadcasted matmul -> softmax(axis=1) -> weighted sum over axis 1."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'MultAttention_10385',
    'batch': 8,        # P: leading dim of vals
    'graphs': 16,      # X: dim 1 of vals, dim 0 of keys_ (softmax / sum axis)
    'groups': 32,      # G: dim 1 of keys_
    'rows': 128,       # I: dim 2 of keys_
    'out_dim': 128,    # O; must equal dim 2 of vals for torch.matmul to be valid
    'key_dim': 256,
    'val_dim': 256,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    ks = jax.random.split(jax.random.key(0), 8)
    vals = jax.random.uniform(ks[0], (c['batch'], c['graphs'], c['out_dim'], c['val_dim']), dtype)
    keys_ = jax.random.uniform(ks[1], (c['graphs'], c['groups'], c['rows'], c['key_dim']), dtype)
    wk = jax.random.normal(ks[2], (c['key_dim'], c['out_dim']), dtype) * 0.02
    bk = jax.random.normal(ks[3], (c['out_dim'],), dtype) * 0.02
    wv = jax.random.normal(ks[4], (c['val_dim'], c['out_dim']), dtype) * 0.02
    bv = jax.random.normal(ks[5], (c['out_dim'],), dtype) * 0.02
    wq = jax.random.normal(ks[6], (c['key_dim'], c['out_dim']), dtype) * 0.02
    bq = jax.random.normal(ks[7], (c['out_dim'],), dtype) * 0.02
    return vals, keys_, wk, bk, wv, bv, wq, bq


def workload(vals, keys_, wk, bk, wv, bv, wq, bq):
    keys = keys_ @ wk + bk        # [X, G, I, O]
    queries = keys_ @ wq + bq     # [X, G, I, O]
    v = vals @ wv + bv            # [P, X, O, O]
    # torch.matmul(keys, v.unsqueeze(1).transpose(1, 2)) with batch broadcasting -> [P, X, G, I, O]
    weights = jnp.einsum('xgik,pxkj->pxgij', keys, v)
    weights = jax.nn.softmax(weights, axis=1)
    summed = jnp.sum(queries[None] * weights, axis=1)  # [P, G, I, O]
    return summed, weights
