"""LogSumExpPool: numerically stable log-mean-exp spatial pooling, (N, C, H, W) -> (N, C, 1, 1)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'LogSumExpPool_4926',
    'batch': 16,
    'channels': 128,
    'height': 128,
    'width': 128,
    'gamma': 4.0,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    feat_map = jax.random.uniform(
        key,
        (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    return (feat_map,)


def workload(feat_map):
    h, w = feat_map.shape[-2], feat_map.shape[-1]
    g = CONFIG['gamma']
    m = jnp.max(feat_map, axis=(-2, -1), keepdims=True)
    area = 1.0 / (h * w)
    s = jnp.sum(jnp.exp(g * (feat_map - m)), axis=(-2, -1), keepdims=True)
    return m + (1.0 / g) * jnp.log(area * s)
