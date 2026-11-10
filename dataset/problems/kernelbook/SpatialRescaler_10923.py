"""[bf16 variant] SpatialRescaler: n_stages of bilinear interpolate (align_corners=False, no antialias) on NCHW input."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'SpatialRescaler_10923',
    'batch': 8,
    'channels': 128,
    'height': 256,
    'width': 384,
    'n_stages': 1,
    'multiplier': 0.5,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    x = jax.random.uniform(key, shape, dtype)
    return (x,)


def _linear_resize_axis(x, axis, multiplier):
    n = x.shape[axis]
    m = int(n * multiplier)
    coords = (jnp.arange(m, dtype=x.dtype) + 0.5) / multiplier - 0.5
    coords = jnp.clip(coords, 0.0, n - 1)
    i0 = jnp.floor(coords).astype(jnp.int32)
    i1 = jnp.minimum(i0 + 1, n - 1)
    weight_shape = [1] * x.ndim
    weight_shape[axis] = m
    w = (coords - i0.astype(x.dtype)).reshape(weight_shape)
    return jnp.take(x, i0, axis=axis) * (1.0 - w) + jnp.take(x, i1, axis=axis) * w


def workload(x):
    multiplier = CONFIG['multiplier']
    for _ in range(CONFIG['n_stages']):
        x = _linear_resize_axis(x, 2, multiplier)
        x = _linear_resize_axis(x, 3, multiplier)
    return x
