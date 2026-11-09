"""BlendConv2d: two Conv2d on x, blended as y0 + (y1 - y0) * t."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'BlendConv2d_697',
    'batch': 8,
    'dim_in': 128,
    'dim_out': 128,
    'height': 130,
    'width': 130,
    'ksize': 3,
    'stride': 1,
    'padding': 0,
    'dilation': 1,
}


def _out_size(n):
    c = CONFIG
    return (n + 2 * c['padding'] - c['dilation'] * (c['ksize'] - 1) - 1) // c['stride'] + 1


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    k_t, k_x, k_w0, k_b0, k_w1, k_b1 = jax.random.split(jax.random.key(0), 6)
    out_h, out_w = _out_size(c['height']), _out_size(c['width'])
    t = jax.random.uniform(k_t, (c['batch'], c['dim_out'], out_h, out_w), dtype)
    x = jax.random.uniform(k_x, (c['batch'], c['dim_in'], c['height'], c['width']), dtype)
    w_shape = (c['ksize'], c['ksize'], c['dim_in'], c['dim_out'])
    w0 = jax.random.normal(k_w0, w_shape, dtype) * 0.02
    b0 = jax.random.normal(k_b0, (c['dim_out'],), dtype) * 0.02
    w1 = jax.random.normal(k_w1, w_shape, dtype) * 0.02
    b1 = jax.random.normal(k_b1, (c['dim_out'],), dtype) * 0.02
    return t, x, w0, b0, w1, b1


def _conv(x_nhwc, w, b):
    c = CONFIG
    p = c['padding']
    y = lax.conv_general_dilated(
        x_nhwc, w,
        window_strides=(c['stride'], c['stride']),
        padding=((p, p), (p, p)),
        rhs_dilation=(c['dilation'], c['dilation']),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
    )
    return y + b


def workload(t, x, w0, b0, w1, b1):
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    y0 = jnp.transpose(_conv(x_nhwc, w0, b0), (0, 3, 1, 2))
    y1 = jnp.transpose(_conv(x_nhwc, w1, b1), (0, 3, 1, 2))
    return y0 + (y1 - y0) * t
