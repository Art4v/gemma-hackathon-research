"""Dual embedding lookup: squeeze(1) -> mu = Embedding(idx), logvar = Embedding(idx)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'ItemInferenceNetwork_12467',
    'batch': 64,
    'seq_len': 2048,
    'num_item': 65536,
    'item_feat_dim': 512,
}


def create_inputs(dtype=jnp.float32):
    k_idx, k_mu, k_lv = jax.random.split(jax.random.key(0), 3)
    shape = (CONFIG['batch'], 1, CONFIG['seq_len'])
    item_index = jax.random.randint(k_idx, shape, 0, CONFIG['num_item'], dtype=jnp.int32)
    table_shape = (CONFIG['num_item'], CONFIG['item_feat_dim'])
    mu_w = jax.random.normal(k_mu, table_shape, dtype) * 0.02
    logvar_w = jax.random.normal(k_lv, table_shape, dtype) * 0.02
    return item_index, mu_w, logvar_w


def workload(item_index, mu_w, logvar_w):
    # torch squeeze(1) is a no-op when dim 1 is not size 1.
    if item_index.shape[1] == 1:
        item_index = jnp.squeeze(item_index, axis=1)
    idx = item_index.astype(jnp.int32)
    mu = jnp.take(mu_w, idx, axis=0)
    logvar = jnp.take(logvar_w, idx, axis=0)
    return mu, logvar
