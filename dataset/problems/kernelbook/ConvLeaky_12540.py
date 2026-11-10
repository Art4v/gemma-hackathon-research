"""Conv2d(3x3) -> LeakyReLU(0.2) -> Conv2d(3x3) -> LeakyReLU(0.2)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'ConvLeaky_12540',
    'batch': 16,
    'in_dim': 128,
    'out_dim': 128,
    'height': 128,
    'width': 128,
    'kernel_size': 3,
    'stride': 1,
    'padding': 1,
    'negative_slope': 0.2,
}


def create_inputs(dtype=jnp.float32):
    k = CONFIG['kernel_size']
    cin, cout = CONFIG['in_dim'], CONFIG['out_dim']
    k_x, k_w1, k_b1, k_w2, k_b2 = jax.random.split(jax.random.key(0), 5)
    x = jax.random.uniform(
        k_x, (CONFIG['batch'], cin, CONFIG['height'], CONFIG['width']), dtype)
    w1 = jax.random.normal(k_w1, (k, k, cin, cout), dtype) * 0.02
    b1 = jax.random.normal(k_b1, (cout,), dtype) * 0.02
    w2 = jax.random.normal(k_w2, (k, k, cout, cout), dtype) * 0.02
    b2 = jax.random.normal(k_b2, (cout,), dtype) * 0.02
    return x, w1, b1, w2, b2


def _conv_leaky(x_nhwc, w_hwio, b):
    s, p = CONFIG['stride'], CONFIG['padding']
    y = lax.conv_general_dilated(
        x_nhwc, w_hwio,
        window_strides=(s, s),
        padding=((p, p), (p, p)),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
    ) + b
    return jnp.where(y >= 0, y, y * CONFIG['negative_slope'])


def workload(x, w1, b1, w2, b2):
    h = jnp.transpose(x, (0, 2, 3, 1))
    h = _conv_leaky(h, w1, b1)
    h = _conv_leaky(h, w2, b2)
    return jnp.transpose(h, (0, 3, 1, 2))
