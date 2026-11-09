"""GELayerv1: AvgPool2d(15, stride 8) -> bilinear upsample (align_corners=True) -> y*x -> x + y."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'GELayerv1_14340',
    'batch': 8,
    'channels': 128,
    'height': 256,
    'width': 256,
    'pool_kernel': 15,
    'pool_stride': 8,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key, (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']), dtype)
    return (x,)


def _interp_axis(y, out_size, axis):
    # Linear interpolation along one axis with align_corners=True semantics.
    in_size = y.shape[axis]
    scale = (in_size - 1) / (out_size - 1) if out_size > 1 else 0.0
    src = jnp.arange(out_size, dtype=jnp.float32) * scale
    i0 = jnp.clip(jnp.floor(src).astype(jnp.int32), 0, in_size - 1)
    i1 = jnp.minimum(i0 + 1, in_size - 1)
    frac = (src - i0.astype(jnp.float32)).astype(y.dtype)
    bshape = [1] * y.ndim
    bshape[axis] = out_size
    frac = frac.reshape(bshape)
    a = jnp.take(y, i0, axis=axis)
    b = jnp.take(y, i1, axis=axis)
    return (1 - frac) * a + frac * b


def workload(x):
    _, _, h, w = x.shape
    k = CONFIG['pool_kernel']
    s = CONFIG['pool_stride']
    pooled = lax.reduce_window(
        x, jnp.array(0, x.dtype), lax.add,
        window_dimensions=(1, 1, k, k),
        window_strides=(1, 1, s, s),
        padding='VALID',
    ) / (k * k)
    y = _interp_axis(pooled, w, axis=3)
    y = _interp_axis(y, h, axis=2)
    return x + y * x
