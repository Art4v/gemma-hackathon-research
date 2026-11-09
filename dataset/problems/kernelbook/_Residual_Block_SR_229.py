"""[bf16 variant] _Residual_Block_SR_229: conv3x3 -> LeakyReLU(0.2) -> conv3x3 -> residual add."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '_Residual_Block_SR_229',
    'batch': 8,
    'num_ft': 128,
    'height': 128,
    'width': 128,
    'kernel_size': 3,
    'padding': 1,
    'negative_slope': 0.2,
}


def create_inputs(dtype=jnp.bfloat16):
    c, k = CONFIG['num_ft'], CONFIG['kernel_size']
    kx, kw1, kb1, kw2, kb2 = jax.random.split(jax.random.key(0), 5)
    x = jax.random.uniform(kx, (CONFIG['batch'], c, CONFIG['height'], CONFIG['width']), dtype)
    w1 = jax.random.normal(kw1, (c, c, k, k), dtype) * 0.02
    b1 = jax.random.normal(kb1, (c,), dtype) * 0.02
    w2 = jax.random.normal(kw2, (c, c, k, k), dtype) * 0.02
    b2 = jax.random.normal(kb2, (c,), dtype) * 0.02
    return x, w1, b1, w2, b2


def _conv(x_nhwc, w_oihw, b):
    p = CONFIG['padding']
    y = lax.conv_general_dilated(
        x_nhwc, w_oihw, window_strides=(1, 1), padding=((p, p), (p, p)),
        dimension_numbers=('NHWC', 'OIHW', 'NHWC'))
    return y + b


def workload(x, w1, b1, w2, b2):
    h = jnp.transpose(x, (0, 2, 3, 1))
    out = jax.nn.leaky_relu(_conv(h, w1, b1), CONFIG['negative_slope'])
    out = _conv(out, w2, b2) + h
    return jnp.transpose(out, (0, 3, 1, 2))
