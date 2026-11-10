"""Conv2d -> ReLU -> Conv2d -> ReLU -> flatten(-1, 512) -> Linear -> ReLU -> Linear (CNN_1870)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'CNN_1870',
    'batch': 16,
    'in_channels': 1,
    'height': 1024,
    'width': 1024,
    'conv1_out': 16,
    'conv1_kernel': 8,
    'conv1_stride': 4,
    'conv2_out': 32,
    'conv2_kernel': 4,
    'conv2_stride': 4,
    'padding': 'VALID',
    'fc1_in': 512,
    'fc1_out': 256,
    'fc2_out': 4,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    keys = jax.random.split(jax.random.key(0), 9)
    x = jax.random.uniform(keys[0], (c['batch'], c['in_channels'], c['height'], c['width']), dtype)
    conv1_w = jax.random.normal(
        keys[1], (c['conv1_out'], c['in_channels'], c['conv1_kernel'], c['conv1_kernel']), dtype
    ) * c['weight_scale']
    conv1_b = jax.random.normal(keys[2], (c['conv1_out'],), dtype) * c['weight_scale']
    conv2_w = jax.random.normal(
        keys[3], (c['conv2_out'], c['conv1_out'], c['conv2_kernel'], c['conv2_kernel']), dtype
    ) * c['weight_scale']
    conv2_b = jax.random.normal(keys[4], (c['conv2_out'],), dtype) * c['weight_scale']
    fc1_w = jax.random.normal(keys[5], (c['fc1_in'], c['fc1_out']), dtype) * c['weight_scale']
    fc1_b = jax.random.normal(keys[6], (c['fc1_out'],), dtype) * c['weight_scale']
    fc2_w = jax.random.normal(keys[7], (c['fc1_out'], c['fc2_out']), dtype) * c['weight_scale']
    fc2_b = jax.random.normal(keys[8], (c['fc2_out'],), dtype) * c['weight_scale']
    return (x, conv1_w, conv1_b, conv2_w, conv2_b, fc1_w, fc1_b, fc2_w, fc2_b)


def workload(x, conv1_w, conv1_b, conv2_w, conv2_b, fc1_w, fc1_b, fc2_w, fc2_b):
    c = CONFIG
    dims = ('NHWC', 'OIHW', 'NHWC')
    y = jnp.transpose(x, (0, 2, 3, 1))
    y = lax.conv_general_dilated(
        y,
        conv1_w,
        window_strides=(c['conv1_stride'], c['conv1_stride']),
        padding=c['padding'],
        dimension_numbers=dims,
    )
    y = jax.nn.relu(y + conv1_b)
    y = lax.conv_general_dilated(
        y,
        conv2_w,
        window_strides=(c['conv2_stride'], c['conv2_stride']),
        padding=c['padding'],
        dimension_numbers=dims,
    )
    y = jax.nn.relu(y + conv2_b)
    y = jnp.transpose(y, (0, 3, 1, 2))
    y = jnp.reshape(y, (-1, fc1_w.shape[0]))
    y = jax.nn.relu(y @ fc1_w + fc1_b)
    return y @ fc2_w + fc2_b
