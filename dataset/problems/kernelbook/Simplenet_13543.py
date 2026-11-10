"""Simplenet: (Conv2d -> ReLU -> MaxPool2d) x2 -> flatten -> (Linear -> ReLU) x2 -> Linear."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'Simplenet_13543',
    'batch': 8192,
    'in_channels': 3,
    'height': 32,
    'width': 32,
    'conv1_out': 6,
    'conv2_out': 16,
    'kernel_size': 5,
    'conv_stride': 1,
    'pool_window': 2,
    'pool_stride': 2,
    'flat_features': 16 * 5 * 5,
    'fc1_out': 120,
    'fc2_out': 84,
    'fc3_out': 10,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    k = c['kernel_size']
    param_shapes = [
        (c['conv1_out'], c['in_channels'], k, k),
        (c['conv1_out'],),
        (c['conv2_out'], c['conv1_out'], k, k),
        (c['conv2_out'],),
        (c['flat_features'], c['fc1_out']),
        (c['fc1_out'],),
        (c['fc1_out'], c['fc2_out']),
        (c['fc2_out'],),
        (c['fc2_out'], c['fc3_out']),
        (c['fc3_out'],),
    ]
    keys = jax.random.split(jax.random.key(0), len(param_shapes) + 1)
    x = jax.random.uniform(
        keys[0],
        (c['batch'], c['in_channels'], c['height'], c['width']),
        dtype,
    )
    params = tuple(
        jax.random.normal(key, shape, dtype) * c['weight_scale']
        for key, shape in zip(keys[1:], param_shapes)
    )
    return (x,) + params


def _conv(x, w, b):
    s = CONFIG['conv_stride']
    y = lax.conv_general_dilated(
        x,
        w,
        window_strides=(s, s),
        padding='VALID',
        dimension_numbers=('NHWC', 'OIHW', 'NHWC'),
    )
    return y + b


def _pool(x):
    k = CONFIG['pool_window']
    s = CONFIG['pool_stride']
    return lax.reduce_window(x, -jnp.inf, lax.max, (1, k, k, 1), (1, s, s, 1), 'VALID')


def workload(x, conv1_w, conv1_b, conv2_w, conv2_b, fc1_w, fc1_b, fc2_w, fc2_b, fc3_w, fc3_b):
    y = jnp.transpose(x, (0, 2, 3, 1))
    y = _pool(jax.nn.relu(_conv(y, conv1_w, conv1_b)))
    y = _pool(jax.nn.relu(_conv(y, conv2_w, conv2_b)))
    y = jnp.transpose(y, (0, 3, 1, 2))
    y = jnp.reshape(y, (-1, CONFIG['flat_features']))
    y = jax.nn.relu(y @ fc1_w + fc1_b)
    y = jax.nn.relu(y @ fc2_w + fc2_b)
    return y @ fc3_w + fc3_b
