"""[bf16 variant] FFB_1689: 1x1 Conv2d(x) + nearest-Upsample(1x1 Conv2d(y)) feature fusion."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'FFB_1689',
    'batch': 8,
    'c1': 128,
    'c2': 128,
    'height': 256,
    'width': 256,
    'y_height': 128,
    'y_width': 128,
    'kernel_size': 1,
    'stride': 1,
    'padding': 'VALID',
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.bfloat16):
    keys = jax.random.split(jax.random.key(0), 6)
    b, c1, c2 = CONFIG['batch'], CONFIG['c1'], CONFIG['c2']
    k = CONFIG['kernel_size']
    scale = CONFIG['weight_scale']
    x = jax.random.uniform(keys[0], (b, c1, CONFIG['height'], CONFIG['width']), dtype)
    y = jax.random.uniform(keys[1], (b, c2, CONFIG['y_height'], CONFIG['y_width']), dtype)
    w1 = jax.random.normal(keys[2], (c1, c1, k, k), dtype) * scale
    b1 = jax.random.normal(keys[3], (c1,), dtype) * scale
    w2 = jax.random.normal(keys[4], (c1, c2, k, k), dtype) * scale
    b2 = jax.random.normal(keys[5], (c1,), dtype) * scale
    return x, y, w1, b1, w2, b2


def _conv(x_nhwc, w, b):
    s = CONFIG['stride']
    out = lax.conv_general_dilated(
        x_nhwc,
        w,
        window_strides=(s, s),
        padding=CONFIG['padding'],
        dimension_numbers=('NHWC', 'OIHW', 'NHWC'),
    )
    return out + b


def _nearest(x_nhwc, out_h, out_w):
    in_h, in_w = x_nhwc.shape[1], x_nhwc.shape[2]
    rows = (jnp.arange(out_h) * in_h) // out_h
    cols = (jnp.arange(out_w) * in_w) // out_w
    return jnp.take(jnp.take(x_nhwc, rows, axis=1), cols, axis=2)


def workload(x, y, w1, b1, w2, b2):
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    y_nhwc = jnp.transpose(y, (0, 2, 3, 1))
    y1 = _conv(x_nhwc, w1, b1)
    y2 = _conv(y_nhwc, w2, b2)
    y2 = _nearest(y2, x_nhwc.shape[1], x_nhwc.shape[2])
    return jnp.transpose(y1 + y2, (0, 3, 1, 2))
