"""GridPredictionModel: Conv2d(k=3, pad=2) + bias -> ReLU -> Conv2d(k=1) + bias -> sigmoid -> spatial crop."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'GridPredictionModel_12942',
    'batch': 8,
    'in_channels': 1,
    'height': 1280,
    'width': 1280,
    'hidden_channels': 100,
    'out_channels': 1,
    'conv1_kernel_size': 3,
    'conv1_padding': 2,
    'conv2_kernel_size': 1,
    'conv2_padding': 0,
    'stride': 1,
    'crop': 1,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    k_x, k_w1, k_b1, k_w2, k_b2 = jax.random.split(key, 5)
    x = jax.random.uniform(
        k_x,
        (CONFIG['batch'], CONFIG['in_channels'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    w1 = jax.random.normal(
        k_w1,
        (
            CONFIG['hidden_channels'],
            CONFIG['in_channels'],
            CONFIG['conv1_kernel_size'],
            CONFIG['conv1_kernel_size'],
        ),
        dtype,
    ) * 0.02
    b1 = jax.random.normal(k_b1, (CONFIG['hidden_channels'],), dtype) * 0.02
    w2 = jax.random.normal(
        k_w2,
        (
            CONFIG['out_channels'],
            CONFIG['hidden_channels'],
            CONFIG['conv2_kernel_size'],
            CONFIG['conv2_kernel_size'],
        ),
        dtype,
    ) * 0.02
    b2 = jax.random.normal(k_b2, (CONFIG['out_channels'],), dtype) * 0.02
    return x, w1, b1, w2, b2


def workload(x, w1, b1, w2, b2):
    strides = (CONFIG['stride'], CONFIG['stride'])
    dims = ('NHWC', 'OIHW', 'NHWC')
    p1 = CONFIG['conv1_padding']
    p2 = CONFIG['conv2_padding']
    c = CONFIG['crop']

    h = jnp.transpose(x, (0, 2, 3, 1))
    h = lax.conv_general_dilated(
        h, w1, window_strides=strides, padding=[(p1, p1), (p1, p1)], dimension_numbers=dims
    ) + b1
    h = jax.nn.relu(h)
    h = lax.conv_general_dilated(
        h, w2, window_strides=strides, padding=[(p2, p2), (p2, p2)], dimension_numbers=dims
    ) + b2
    h = jax.nn.sigmoid(h)
    h = jnp.transpose(h, (0, 3, 1, 2))
    return h[:, :, c:-c, c:-c]
