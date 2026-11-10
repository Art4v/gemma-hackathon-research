"""[bf16 variant] UNet_3564: (conv-lrelu x2, maxpool) x4, conv-lrelu x2, (conv_transpose, skip concat, conv-lrelu x2) x4, conv, pixel_shuffle."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'UNet_3564',
    'batch': 8,
    'in_channels': 3,
    'height': 768,
    'width': 768,
    'encoder_channels': (32, 64, 128, 256, 512),
    'final_channels': 12,
    'kernel_size': 3,
    'stride': 1,
    'padding': 1,
    'pool_size': 2,
    'up_kernel_size': 2,
    'up_stride': 2,
    'up_padding': 0,
    'lrelu_slope': 0.2,
    'upscale': 2,
    'weight_scale': 0.02,
}


def _param_shapes():
    k = CONFIG['kernel_size']
    ku = CONFIG['up_kernel_size']
    channels = CONFIG['encoder_channels']

    def conv(c_in, c_out):
        return [(c_out, c_in, k, k), (c_out,)]

    def upconv(c_in, c_out):
        return [(c_in, c_out, ku, ku), (c_out,)]

    shapes = []
    prev = CONFIG['in_channels']
    for c in channels:
        shapes += conv(prev, c) + conv(c, c)
        prev = c
    for c in reversed(channels[:-1]):
        shapes += upconv(prev, c) + conv(2 * c, c) + conv(c, c)
        prev = c
    shapes += conv(prev, CONFIG['final_channels'])
    return shapes


def create_inputs(dtype=jnp.bfloat16):
    shapes = _param_shapes()
    keys = jax.random.split(jax.random.key(0), len(shapes) + 1)
    x = jax.random.uniform(
        keys[0],
        (CONFIG['batch'], CONFIG['in_channels'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    params = tuple(
        jax.random.normal(k, s, dtype) * CONFIG['weight_scale']
        for k, s in zip(keys[1:], shapes)
    )
    return (x,) + params


def _lrelu(x):
    return jax.nn.leaky_relu(x, negative_slope=CONFIG['lrelu_slope'])


def _conv(x, w, b):
    s = CONFIG['stride']
    p = CONFIG['padding']
    y = lax.conv_general_dilated(
        x,
        w,
        window_strides=(s, s),
        padding=[(p, p), (p, p)],
        dimension_numbers=('NHWC', 'OIHW', 'NHWC'),
    )
    return y + b


def _conv_transpose(x, w, b):
    s = CONFIG['up_stride']
    q = CONFIG['up_kernel_size'] - 1 - CONFIG['up_padding']
    y = lax.conv_general_dilated(
        x,
        w[:, :, ::-1, ::-1],
        window_strides=(1, 1),
        padding=[(q, q), (q, q)],
        lhs_dilation=(s, s),
        dimension_numbers=('NHWC', 'IOHW', 'NHWC'),
    )
    return y + b


def _max_pool(x):
    p = CONFIG['pool_size']
    return lax.reduce_window(x, -jnp.inf, lax.max, (1, p, p, 1), (1, p, p, 1), 'VALID')


def _pixel_shuffle(x, r):
    b, c, h, w = x.shape
    oc = c // (r * r)
    y = jnp.reshape(x, (b, oc, r, r, h, w))
    y = jnp.transpose(y, (0, 1, 4, 2, 5, 3))
    return jnp.reshape(y, (b, oc, h * r, w * r))


def workload(*inputs):
    params = iter(inputs[1:])

    def conv_block(h):
        h = _lrelu(_conv(h, next(params), next(params)))
        h = _lrelu(_conv(h, next(params), next(params)))
        return h

    h = jnp.transpose(inputs[0], (0, 2, 3, 1))

    levels = len(CONFIG['encoder_channels'])
    skips = []
    for level in range(levels):
        h = conv_block(h)
        if level < levels - 1:
            skips.append(h)
            h = _max_pool(h)

    for skip in reversed(skips):
        up = _conv_transpose(h, next(params), next(params))
        h = jnp.concatenate([skip, up], axis=-1)
        h = conv_block(h)

    h = _conv(h, next(params), next(params))
    out = jnp.transpose(h, (0, 3, 1, 2))
    return _pixel_shuffle(out, CONFIG['upscale'])
