"""[bf16 variant] ResidualBlock (R2D3/IMPALA): relu, bias, conv3x3, bias, relu, bias, conv3x3, scale, bias, residual add."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'ResidualBlock_8570',
    'batch': 8,
    'channels': 128,
    'height': 128,
    'width': 128,
    'kernel_size': 3,
    'stride': 1,
    'padding': 1,
    'weight_std': 0.02,
}


def create_inputs(dtype=jnp.bfloat16):
    batch = CONFIG['batch']
    channels = CONFIG['channels']
    height = CONFIG['height']
    width = CONFIG['width']
    kernel_size = CONFIG['kernel_size']
    weight_std = CONFIG['weight_std']

    key = jax.random.key(0)
    k_x, k_w1, k_w2 = jax.random.split(key, 3)

    x = jax.random.uniform(k_x, (batch, channels, height, width), dtype)
    weight_shape = (channels, channels, kernel_size, kernel_size)
    w1 = jax.random.normal(k_w1, weight_shape, dtype) * weight_std
    w2 = jax.random.normal(k_w2, weight_shape, dtype) * weight_std

    param_shape = (channels, 1, 1)
    bias1 = jnp.zeros(param_shape, dtype)
    bias2 = jnp.zeros(param_shape, dtype)
    bias3 = jnp.zeros(param_shape, dtype)
    bias4 = jnp.zeros(param_shape, dtype)
    scale = jnp.ones(param_shape, dtype)

    return x, w1, w2, bias1, bias2, bias3, bias4, scale


def _conv(x, w):
    stride = CONFIG['stride']
    padding = CONFIG['padding']
    y = jnp.transpose(x, (0, 2, 3, 1))
    y = lax.conv_general_dilated(
        y,
        w,
        window_strides=(stride, stride),
        padding=[(padding, padding), (padding, padding)],
        dimension_numbers=('NHWC', 'OIHW', 'NHWC'),
    )
    return jnp.transpose(y, (0, 3, 1, 2))


def workload(x, w1, w2, bias1, bias2, bias3, bias4, scale):
    x = jax.nn.relu(x)
    original = x
    x = x + bias1
    x = _conv(x, w1)
    x = x + bias2
    x = jax.nn.relu(x)
    x = x + bias3
    x = _conv(x, w2)
    x = x * scale
    x = x + bias4
    return original + x
