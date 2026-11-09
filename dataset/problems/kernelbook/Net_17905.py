"""Conv5x5+ReLU -> 3x(Conv3x3+ReLU) -> Conv3x3+ReLU -> Conv3x3+ReLU -> PixelShuffle -> Conv1x1 (super-resolution net)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'Net_17905',
    'batch': 16,
    'in_channels': 1,
    'height': 1024,
    'width': 1024,
    'hidden': 64,
    'mid': 32,
    'upscale_factor': 1,
    'k1': 5,
    'k': 3,
    'k5': 1,
}


def _conv_shapes():
    c = CONFIG
    r2 = c['upscale_factor'] ** 2
    return [
        (c['k1'], c['k1'], c['in_channels'], c['hidden']),
        (c['k'], c['k'], c['hidden'], c['hidden']),
        (c['k'], c['k'], c['hidden'], c['hidden']),
        (c['k'], c['k'], c['hidden'], c['hidden']),
        (c['k'], c['k'], c['hidden'], c['mid']),
        (c['k'], c['k'], c['mid'], r2),
        (c['k5'], c['k5'], 1, 1),
    ]


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    shapes = _conv_shapes()
    keys = jax.random.split(jax.random.key(0), 1 + 2 * len(shapes))
    x = jax.random.uniform(keys[0], (c['batch'], c['in_channels'], c['height'], c['width']), dtype)
    params = []
    for i, s in enumerate(shapes):
        params.append(jax.random.normal(keys[1 + 2 * i], s, dtype) * 0.02)
        params.append(jax.random.normal(keys[2 + 2 * i], (s[3],), dtype) * 0.02)
    return (x, *params)


def _conv(x, w, b):
    # 'SAME' padding matches PyTorch's padding=k//2 for odd kernels (5->2, 3->1, 1->0).
    y = lax.conv_general_dilated(
        x, w, window_strides=(1, 1), padding='SAME',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
    return y + b


def _pixel_shuffle_nhwc(x, r):
    n, h, w, crr = x.shape
    c = crr // (r * r)
    x = x.reshape(n, h, w, c, r, r)
    x = jnp.transpose(x, (0, 1, 4, 2, 5, 3))
    return x.reshape(n, h * r, w * r, c)


def workload(x, w1, b1, w21, b21, w22, b22, w23, b23, w3, b3, w4, b4, w5, b5):
    r = CONFIG['upscale_factor']
    out = jnp.transpose(x, (0, 2, 3, 1))
    for w, b in ((w1, b1), (w21, b21), (w22, b22), (w23, b23), (w3, b3), (w4, b4)):
        out = jax.nn.relu(_conv(out, w, b))
    out = _pixel_shuffle_nhwc(out, r)
    out = _conv(out, w5, b5)
    return jnp.transpose(out, (0, 3, 1, 2))
