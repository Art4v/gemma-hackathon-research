"""OuterProductLayer (PNN): pairwise field gather -> kernel-weighted outer product -> sum."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'OuterProductLayer_1919',
    'batch': 2048,
    'num_feature_field': 32,
    'embedding_size': 128,
}


def create_inputs(dtype=jnp.float32):
    b = CONFIG['batch']
    f = CONFIG['num_feature_field']
    e = CONFIG['embedding_size']
    num_pairs = f * (f - 1) // 2
    k_x, k_w = jax.random.split(jax.random.key(0))
    feat_emb = jax.random.uniform(k_x, (b, f, e), dtype)
    kernel = jax.random.normal(k_w, (e, num_pairs, e), dtype) * 0.02
    return feat_emb, kernel


def workload(feat_emb, kernel):
    num_fields = feat_emb.shape[1]
    # Row-major i<j order, matching the nested loops in the PyTorch module.
    row, col = jnp.triu_indices(num_fields, k=1)
    p = feat_emb[:, row]  # (B, P, E)
    q = feat_emb[:, col]  # (B, P, E)
    # (B, P, E_out): sum over last embed axis of p * kernel, then transposed.
    pk = jnp.einsum('bkf,ekf->bke', p, kernel, precision=lax.Precision.HIGHEST)
    return jnp.sum(pk * q, axis=-1)  # (B, P)
