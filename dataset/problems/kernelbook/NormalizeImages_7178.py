"""NormalizeImages: per-sample (x - mean) / (unbiased std + eps) over all non-batch dims."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'NormalizeImages_7178',
    'batch': 16,
    'channels': 8,
    'height': 512,
    'width': 512,
    'eps': 1e-07,
    'ddof': 1,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key, (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']), dtype
    )
    return (x,)


def workload(x):
    flat = x.reshape(x.shape[0], -1)
    mp = jnp.mean(flat, axis=1)
    sp = jnp.std(flat, axis=1, ddof=CONFIG['ddof']) + CONFIG['eps']
    bshape = (x.shape[0],) + (1,) * (x.ndim - 1)
    return (x - mp.reshape(bshape)) / sp.reshape(bshape)
