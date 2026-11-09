"""[bf16 variant] AvgPool2d(2) -> Conv2d(4x4, pad 1) + LeakyReLU(0.1) -> Conv2d(4x4, pad 1) + LeakyReLU(0.1)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'down_17191',
    'batch': 8,
    'in_channels': 128,
    'out_channels': 128,
    'height': 256,
    'width': 256,
    'filter_size': 4,
    'pool': 2,
    'negative_slope': 0.1,
}


def create_inputs(dtype=jnp.bfloat16):
    k = jax.random.split(jax.random.key(0), 5)
    c = CONFIG
    f, cin, cout = c['filter_size'], c['in_channels'], c['out_channels']
    x = jax.random.uniform(k[0], (c['batch'], cin, c['height'], c['width']), dtype)
    w1 = jax.random.normal(k[1], (f, f, cin, cout), dtype) * 0.02
    b1 = jax.random.normal(k[2], (cout,), dtype) * 0.02
    w2 = jax.random.normal(k[3], (f, f, cout, cout), dtype) * 0.02
    b2 = jax.random.normal(k[4], (cout,), dtype) * 0.02
    return x, w1, b1, w2, b2


def _conv_lrelu(x, w, b):
    pad = (CONFIG['filter_size'] - 1) // 2  # PyTorch: int((f-1)/2), symmetric
    y = lax.conv_general_dilated(
        x, w, window_strides=(1, 1), padding=((pad, pad), (pad, pad)),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
    y = y + b
    return jnp.where(y >= 0, y, CONFIG['negative_slope'] * y)


def workload(x, w1, b1, w2, b2):
    p = CONFIG['pool']
    x = jnp.transpose(x, (0, 2, 3, 1))
    n, h, w, c = x.shape
    ho, wo = h // p, w // p
    x = x[:, :ho * p, :wo * p, :].reshape(n, ho, p, wo, p, c).mean(axis=(2, 4))
    x = _conv_lrelu(x, w1, b1)
    x = _conv_lrelu(x, w2, b2)
    return jnp.transpose(x, (0, 3, 1, 2))
