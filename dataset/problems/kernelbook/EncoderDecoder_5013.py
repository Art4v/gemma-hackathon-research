"""adaptive_max_pool2d(H/2, W/2) -> bilinear interpolate to (H, W) (align_corners=False) -> sigmoid."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'EncoderDecoder_5013',
    'batch': 16,
    'channels': 64,
    'height': 256,
    'width': 256,
    'pool_factor': 2,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']),
        dtype=dtype,
    )
    return (x,)


def _bilinear_axis(x, out_size, axis):
    # PyTorch align_corners=False: half-pixel source coords, clamped at 0 on the left.
    in_size = x.shape[axis]
    scale = in_size / out_size
    src = (jnp.arange(out_size, dtype=jnp.float32) + 0.5) * scale - 0.5
    src = jnp.maximum(src, 0.0)
    i0 = jnp.floor(src).astype(jnp.int32)
    i1 = jnp.minimum(i0 + 1, in_size - 1)
    lam = (src - i0.astype(jnp.float32)).astype(x.dtype)
    shape = [1] * x.ndim
    shape[axis] = out_size
    lam = lam.reshape(shape)
    a = jnp.take(x, i0, axis=axis)
    b = jnp.take(x, i1, axis=axis)
    return a * (1 - lam) + b * lam


def workload(x):
    b, c, h, w = x.shape
    f = CONFIG['pool_factor']
    # ponytail: adaptive pool to (h//2, w//2) with even h, w is exactly a 2x2/stride-2 max pool.
    pooled = x.reshape(b, c, h // f, f, w // f, f).max(axis=(3, 5))
    up = _bilinear_axis(pooled, h, axis=2)
    up = _bilinear_axis(up, w, axis=3)
    return jax.nn.sigmoid(up)
