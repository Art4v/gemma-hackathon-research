"""NetEnd_17238: 1x1 Conv2d with bias followed by channel-axis Softmax."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'NetEnd_17238',
    'batch': 8,
    'in_channels': 21,
    'height': 512,
    'width': 512,
    'num_classes': 4,
    'kernel_size': 1,
    'stride': 1,
    'padding': 0,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    k_x, k_w, k_b = jax.random.split(key, 3)
    x = jax.random.uniform(
        k_x,
        (CONFIG['batch'], CONFIG['in_channels'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    w = jax.random.normal(
        k_w,
        (CONFIG['num_classes'], CONFIG['in_channels'], CONFIG['kernel_size'], CONFIG['kernel_size']),
        dtype,
    ) * 0.02
    b = jax.random.normal(k_b, (CONFIG['num_classes'],), dtype) * 0.02
    return x, w, b


def workload(x, w, b):
    stride = CONFIG['stride']
    pad = CONFIG['padding']
    y = jnp.transpose(x, (0, 2, 3, 1))
    y = lax.conv_general_dilated(
        y,
        w,
        window_strides=(stride, stride),
        padding=[(pad, pad), (pad, pad)],
        dimension_numbers=('NHWC', 'OIHW', 'NHWC'),
    )
    y = y + b
    y = jax.nn.softmax(y, axis=-1)
    return jnp.transpose(y, (0, 3, 1, 2))
