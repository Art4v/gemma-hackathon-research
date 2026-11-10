"""BatchSpectralShrinkage: transpose -> SVD singular values -> sum of squares of the k smallest."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'BatchSpectralShrinkage_5680',
    'batch': 2048,
    'feature_dim': 8192,
    'k': 1,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    feature = jax.random.uniform(key, (CONFIG['batch'], CONFIG['feature_dim']), dtype)
    return (feature,)


def workload(feature):
    s = jnp.linalg.svd(feature.T, full_matrices=False, compute_uv=False)
    num = s.shape[0]
    return jnp.sum(jnp.square(s[num - CONFIG['k']:]))
