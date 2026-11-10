"""Upsample4x: bilinear 4x upsample (align_corners=False) -> Conv2d 3x3 with bias -> ReLU."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'Upsample4x_4595',
    'batch': 8,
    'n_channels': 128,
    'height': 128,
    'width': 128,
    'scale_factor': 4,
    'kernel_size': 3,
    'stride': 1,
    'padding': 1,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    k_x, k_kernel, k_bias = jax.random.split(key, 3)
    c = CONFIG['n_channels']
    ks = CONFIG['kernel_size']
    x = jax.random.uniform(
        k_x, (CONFIG['batch'], c, CONFIG['height'], CONFIG['width']), dtype
    )
    kernel = jax.random.normal(k_kernel, (c, c, ks, ks), dtype) * 0.02
    bias = jax.random.normal(k_bias, (c,), dtype) * 0.02
    return x, kernel, bias


def workload(x, kernel, bias):
    scale = CONFIG['scale_factor']
    stride = CONFIG['stride']
    pad = CONFIG['padding']

    y = jnp.transpose(x, (0, 2, 3, 1))

    for axis in (1, 2):
        n = y.shape[axis]
        out = n * scale
        coord = (jnp.arange(out, dtype=y.dtype) + 0.5) / scale - 0.5
        floor = jnp.floor(coord)
        shape = [1] * y.ndim
        shape[axis] = out
        lam = (coord - floor).reshape(shape)
        i0 = floor.astype(jnp.int32)
        lo = jnp.clip(i0, 0, n - 1)
        hi = jnp.clip(i0 + 1, 0, n - 1)
        y = jnp.take(y, lo, axis=axis) * (1 - lam) + jnp.take(y, hi, axis=axis) * lam

    y = lax.conv_general_dilated(
        y,
        kernel,
        window_strides=(stride, stride),
        padding=[(pad, pad), (pad, pad)],
        dimension_numbers=('NHWC', 'OIHW', 'NHWC'),
    )
    y = y + bias
    y = jnp.maximum(y, 0)
    return jnp.transpose(y, (0, 3, 1, 2))
