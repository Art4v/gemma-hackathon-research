"""Four 1x1 Conv2d + bias + ReLU layers followed by LogSoftmax over channels (net2)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'net2_7139',
    'batch': 8,
    'in_channels': 4,
    'hidden_channels': (64, 256, 128),
    'n_classes': 2,
    'height': 640,
    'width': 640,
    'kernel_size': 1,
    'stride': 1,
    'padding': 0,
    'dilation': 1,
    'channel_axis': 1,
    'init_scale': 0.02,
}


def create_inputs(dtype=jnp.float32):
    channels = (CONFIG['in_channels'],) + tuple(CONFIG['hidden_channels']) + (CONFIG['n_classes'],)
    n_layers = len(channels) - 1
    keys = jax.random.split(jax.random.key(0), 1 + 2 * n_layers)
    x = jax.random.uniform(
        keys[0],
        (CONFIG['batch'], CONFIG['in_channels'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    params = []
    k = CONFIG['kernel_size']
    for i in range(n_layers):
        w = jax.random.normal(keys[1 + 2 * i], (channels[i + 1], channels[i], k, k), dtype) * CONFIG['init_scale']
        b = jax.random.normal(keys[2 + 2 * i], (channels[i + 1],), dtype) * CONFIG['init_scale']
        params.extend([w, b])
    return (x,) + tuple(params)


def _conv(x, w, b):
    s = CONFIG['stride']
    p = CONFIG['padding']
    d = CONFIG['dilation']
    y = lax.conv_general_dilated(
        x,
        w,
        window_strides=(s, s),
        padding=[(p, p), (p, p)],
        rhs_dilation=(d, d),
        dimension_numbers=('NHWC', 'OIHW', 'NHWC'),
    )
    return y + b


def workload(x, w1, b1, w2, b2, w3, b3, w4, b4):
    h = jnp.transpose(x, (0, 2, 3, 1))
    h = jax.nn.relu(_conv(h, w1, b1))
    h = jax.nn.relu(_conv(h, w2, b2))
    h = jax.nn.relu(_conv(h, w3, b3))
    h = jax.nn.relu(_conv(h, w4, b4))
    h = jnp.transpose(h, (0, 3, 1, 2))
    return jax.nn.log_softmax(h, axis=CONFIG['channel_axis'])
