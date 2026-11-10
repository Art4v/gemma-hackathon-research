"""SEBlock_5517: avg_pool2d -> 1x1 conv -> relu -> 1x1 conv -> sigmoid -> channel-wise rescale of inputs."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'SEBlock_5517',
    'batch': 8,
    'input_channels': 128,
    'internal_neurons': 128,
    'height': 256,
    'width': 256,
    'conv_kernel_size': 1,
    'conv_stride': 1,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.float32):
    keys = jax.random.split(jax.random.key(0), 5)
    c = CONFIG['input_channels']
    n = CONFIG['internal_neurons']
    k = CONFIG['conv_kernel_size']
    scale = CONFIG['weight_scale']
    x = jax.random.uniform(
        keys[0], (CONFIG['batch'], c, CONFIG['height'], CONFIG['width']), dtype
    )
    down_w = jax.random.normal(keys[1], (n, c, k, k), dtype) * scale
    down_b = jax.random.normal(keys[2], (n,), dtype) * scale
    up_w = jax.random.normal(keys[3], (c, n, k, k), dtype) * scale
    up_b = jax.random.normal(keys[4], (c,), dtype) * scale
    return x, down_w, down_b, up_w, up_b


def _conv(x, w, b):
    s = CONFIG['conv_stride']
    y = lax.conv_general_dilated(
        x,
        w,
        window_strides=(s, s),
        padding='VALID',
        dimension_numbers=('NHWC', 'OIHW', 'NHWC'),
    )
    return y + b.reshape((1, 1, 1, -1))


def workload(x, down_w, down_b, up_w, up_b):
    k = x.shape[3]
    pooled = lax.reduce_window(
        x, 0.0, lax.add, (1, 1, k, k), (1, 1, k, k), 'VALID'
    ) / (k * k)
    h = jnp.transpose(pooled, (0, 2, 3, 1))
    h = jax.nn.relu(_conv(h, down_w, down_b))
    h = jax.nn.sigmoid(_conv(h, up_w, up_b))
    scale = jnp.transpose(h, (0, 3, 1, 2)).reshape((-1, x.shape[1], 1, 1))
    return x * scale
