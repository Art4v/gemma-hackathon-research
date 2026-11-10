"""[bf16 variant] ConvBlock_5722: conv2d -> relu -> maxpool2d -> conv2d -> relu -> maxpool2d -> flatten."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'ConvBlock_5722',
    'batch': 32,
    'in_channels': 3,
    'height': 512,
    'width': 512,
    'conv1_channels': 6,
    'conv2_channels': 16,
    'kernel_size': 5,
    'conv_stride': 1,
    'conv_padding': 'VALID',
    'pool_size': 2,
    'pool_stride': 2,
    'pool_padding': 'VALID',
    'flatten_size': 16 * 5 * 5,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.bfloat16):
    kx, kw1, kb1, kw2, kb2 = jax.random.split(jax.random.key(0), 5)
    k = CONFIG['kernel_size']
    c0 = CONFIG['in_channels']
    c1 = CONFIG['conv1_channels']
    c2 = CONFIG['conv2_channels']
    scale = CONFIG['weight_scale']
    x = jax.random.uniform(kx, (CONFIG['batch'], c0, CONFIG['height'], CONFIG['width']), dtype)
    w1 = jax.random.normal(kw1, (c1, c0, k, k), dtype) * scale
    b1 = jax.random.normal(kb1, (c1,), dtype) * scale
    w2 = jax.random.normal(kw2, (c2, c1, k, k), dtype) * scale
    b2 = jax.random.normal(kb2, (c2,), dtype) * scale
    return x, w1, b1, w2, b2


def workload(x, w1, b1, w2, b2):
    conv_stride = (CONFIG['conv_stride'], CONFIG['conv_stride'])
    pool_window = (1, CONFIG['pool_size'], CONFIG['pool_size'], 1)
    pool_strides = (1, CONFIG['pool_stride'], CONFIG['pool_stride'], 1)
    y = jnp.transpose(x, (0, 2, 3, 1))
    for w, b in ((w1, b1), (w2, b2)):
        y = lax.conv_general_dilated(
            y,
            w,
            window_strides=conv_stride,
            padding=CONFIG['conv_padding'],
            dimension_numbers=('NHWC', 'OIHW', 'NHWC'),
        )
        y = y + b
        y = jnp.maximum(y, 0.0)
        y = lax.reduce_window(y, -jnp.inf, lax.max, pool_window, pool_strides, CONFIG['pool_padding'])
    y = jnp.transpose(y, (0, 3, 1, 2))
    return y.reshape(-1, CONFIG['flatten_size'])
