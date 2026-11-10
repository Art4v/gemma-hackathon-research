"""AvgPool2d(2) -> Conv2d + LeakyReLU(0.1) -> Conv2d + LeakyReLU(0.1) (UNet down block)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'down_13437',
    'batch': 8,
    'in_channels': 128,
    'out_channels': 128,
    'height': 256,
    'width': 256,
    'filter_size': 4,
    'pool': 2,
    'negative_slope': 0.1,
}


def create_inputs(dtype=jnp.float32):
    k = jax.random.split(jax.random.key(0), 5)
    n, ci, co = CONFIG['batch'], CONFIG['in_channels'], CONFIG['out_channels']
    fs = CONFIG['filter_size']
    x = jax.random.uniform(k[0], (n, ci, CONFIG['height'], CONFIG['width']), dtype)
    w1 = jax.random.normal(k[1], (fs, fs, ci, co), dtype) * 0.02
    b1 = jax.random.normal(k[2], (co,), dtype) * 0.02
    w2 = jax.random.normal(k[3], (fs, fs, co, co), dtype) * 0.02
    b2 = jax.random.normal(k[4], (co,), dtype) * 0.02
    return x, w1, b1, w2, b2


def _conv_lrelu(x, w, b):
    # PyTorch padding=int((k-1)/2); for even k this shrinks spatial dims by 1.
    ph, pw = (w.shape[0] - 1) // 2, (w.shape[1] - 1) // 2
    y = lax.conv_general_dilated(
        x, w, window_strides=(1, 1), padding=((ph, ph), (pw, pw)),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        precision=lax.Precision.HIGHEST,
    ) + b
    return jax.nn.leaky_relu(y, CONFIG['negative_slope'])


def workload(x, w1, b1, w2, b2):
    p = CONFIG['pool']
    h = jnp.transpose(x, (0, 2, 3, 1))
    h = lax.reduce_window(h, 0.0, lax.add, (1, p, p, 1), (1, p, p, 1), 'VALID') / (p * p)
    h = _conv_lrelu(h, w1, b1)
    h = _conv_lrelu(h, w2, b2)
    return jnp.transpose(h, (0, 3, 1, 2))
