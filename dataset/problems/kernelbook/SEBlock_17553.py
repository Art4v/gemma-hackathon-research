"""SEBlock: global avg_pool2d -> 1x1 conv -> relu -> 1x1 conv -> sigmoid -> multiply input by channel attention."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'SEBlock_17553',
    'batch': 8,
    'channels': 128,
    'height': 256,
    'width': 256,
    'conv_kernel': 1,
    'conv_stride': 1,
    'conv_padding': 'VALID',
    'pool_padding': 'VALID',
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.float32):
    keys = jax.random.split(jax.random.key(0), 5)
    c = CONFIG['channels']
    k = CONFIG['conv_kernel']
    scale = CONFIG['weight_scale']
    x = jax.random.uniform(
        keys[0], (CONFIG['batch'], c, CONFIG['height'], CONFIG['width']), dtype
    )
    w1 = jax.random.normal(keys[1], (c, c, k, k), dtype) * scale
    b1 = jax.random.normal(keys[2], (c,), dtype) * scale
    w2 = jax.random.normal(keys[3], (c, c, k, k), dtype) * scale
    b2 = jax.random.normal(keys[4], (c,), dtype) * scale
    return x, w1, b1, w2, b2


def _conv(h, w, b):
    stride = CONFIG['conv_stride']
    out = lax.conv_general_dilated(
        h,
        w,
        window_strides=(stride, stride),
        padding=CONFIG['conv_padding'],
        dimension_numbers=('NHWC', 'OIHW', 'NHWC'),
    )
    return out + b


def workload(x, w1, b1, w2, b2):
    pool = x.shape[2]
    h = jnp.transpose(x, (0, 2, 3, 1))
    h = lax.reduce_window(
        h,
        0.0,
        lax.add,
        (1, pool, pool, 1),
        (1, pool, pool, 1),
        CONFIG['pool_padding'],
    ) / (pool * pool)
    h = jax.nn.relu(_conv(h, w1, b1))
    h = jax.nn.sigmoid(_conv(h, w2, b2))
    h = jnp.transpose(h, (0, 3, 1, 2))
    return x * h
