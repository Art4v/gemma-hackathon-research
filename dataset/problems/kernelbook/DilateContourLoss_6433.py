"""DilateContourLoss: maxpool3x3 dilation -> clamp(diff) -> sums -> normalize by y_true.sum()."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'DilateContourLoss_6433',
    'batch': 16,
    'channels': 16,
    'height': 512,
    'width': 512,
    'pool_kernel': 3,
    'pool_stride': 1,
    'pool_padding': 1,
}


def create_inputs(dtype=jnp.float32):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    k1, k2 = jax.random.split(jax.random.key(0))
    y_pred = jax.random.uniform(k1, shape, dtype)
    y_true = jax.random.uniform(k2, shape, dtype)
    return y_pred, y_true


def _dilate(x):
    k, s, p = CONFIG['pool_kernel'], CONFIG['pool_stride'], CONFIG['pool_padding']
    return lax.reduce_window(
        x, jnp.array(-jnp.inf, x.dtype), lax.max,
        window_dimensions=(1, 1, k, k),
        window_strides=(1, 1, s, s),
        padding=((0, 0), (0, 0), (p, p), (p, p)),
    )


def workload(y_pred, y_true):
    miss = jnp.clip(y_true - _dilate(y_pred), 0, 1)
    redun = jnp.clip(y_pred - _dilate(y_true), 0, 1)
    return (miss.sum() + redun.sum()) / y_true.sum()
