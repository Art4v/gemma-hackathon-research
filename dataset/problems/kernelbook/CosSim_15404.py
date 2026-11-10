"""CosSim: L2-normalize features and centroids on the last axis, then matmul x @ centroids^T."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'CosSim_15404',
    'batch': 8,
    'height': 128,
    'width': 128,
    'nfeat': 256,
    'nclass': 256,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    k_x, k_c = jax.random.split(key)
    x = jax.random.uniform(
        k_x,
        (CONFIG['batch'], CONFIG['height'], CONFIG['width'], CONFIG['nfeat']),
        dtype,
    )
    centroids = jax.random.normal(k_c, (CONFIG['nclass'], CONFIG['nfeat']), dtype) * 0.02
    return x, centroids


def workload(x, centroids):
    norms = jnp.linalg.norm(x, ord=2, axis=-1, keepdims=True)
    nfeat = x / norms
    norms_c = jnp.linalg.norm(centroids, ord=2, axis=-1, keepdims=True)
    ncenters = centroids / norms_c
    logits = jnp.matmul(nfeat, jnp.swapaxes(ncenters, 0, 1))
    return logits
