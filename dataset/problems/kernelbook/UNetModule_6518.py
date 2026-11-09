"""UNetModule: conv3x3(no bias) -> ReLU -> conv3x3(no bias) -> ReLU."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'UNetModule_6518',
    'batch': 8,
    'in_channels': 128,
    'out_channels': 128,
    'height': 128,
    'width': 128,
    'kernel_size': 3,
    'padding': 1,
}


def create_inputs(dtype=jnp.float32):
    k_x, k_w1, k_w2 = jax.random.split(jax.random.key(0), 3)
    c = CONFIG
    k = c['kernel_size']
    x = jax.random.uniform(
        k_x, (c['batch'], c['in_channels'], c['height'], c['width']), dtype)
    w1 = jax.random.normal(k_w1, (c['out_channels'], c['in_channels'], k, k), dtype) * 0.02
    w2 = jax.random.normal(k_w2, (c['out_channels'], c['out_channels'], k, k), dtype) * 0.02
    return x, w1, w2


def _conv_relu(x_nhwc, w_oihw):
    p = CONFIG['padding']
    y = lax.conv_general_dilated(
        x_nhwc, w_oihw,
        window_strides=(1, 1),
        padding=((p, p), (p, p)),
        dimension_numbers=('NHWC', 'OIHW', 'NHWC'),
    )
    return jax.nn.relu(y)


def workload(x, w1, w2):
    h = jnp.transpose(x, (0, 2, 3, 1))
    h = _conv_relu(h, w1)
    h = _conv_relu(h, w2)
    return jnp.transpose(h, (0, 3, 1, 2))
