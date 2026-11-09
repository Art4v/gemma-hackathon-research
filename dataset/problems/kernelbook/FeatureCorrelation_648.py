"""FeatureCorrelation (3D): transposed-flatten bmm correlation -> ReLU -> L2 norm over correlation channel."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'FeatureCorrelation_648',
    'batch': 8,
    'channels': 256,
    'height': 64,
    'width': 64,
    'eps': 1e-6,
}


def create_inputs(dtype=jnp.float32):
    k_a, k_b = jax.random.split(jax.random.key(0))
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    feature_a = jax.random.uniform(k_a, shape, dtype)
    feature_b = jax.random.uniform(k_b, shape, dtype)
    return feature_a, feature_b


def workload(feature_a, feature_b):
    b, c, h, w = feature_a.shape
    # torch transposes A to (b,c,w,h) before flattening, so A's flat index is w-major.
    a_flat = jnp.transpose(feature_a, (0, 1, 3, 2)).reshape(b, c, h * w)
    b_flat = feature_b.reshape(b, c, h * w)
    corr = jnp.einsum('bck,bcm->bkm', a_flat, b_flat,
                      precision=jax.lax.Precision.HIGHEST)
    corr = corr.reshape(b, h * w, h, w)
    corr = jax.nn.relu(corr)
    norm = jnp.sqrt(jnp.sum(corr * corr, axis=1, keepdims=True) + CONFIG['eps'])
    return corr / norm
