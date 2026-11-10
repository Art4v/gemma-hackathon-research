"""ConvNet_11234: conv-relu-maxpool x2, flatten, then fc-relu-fc-relu-fc."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'ConvNet_11234',
    'batch': 8192,
    'in_channels': 3,
    'height': 32,
    'width': 32,
    'conv1_channels': 6,
    'conv2_channels': 16,
    'kernel_size': 5,
    'pool_size': 2,
    'pool_stride': 2,
    'flat_height': 5,
    'flat_width': 5,
    'flat_features': 16 * 5 * 5,
    'fc1_features': 120,
    'fc2_features': 84,
    'fc3_features': 5,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.float32):
    keys = jax.random.split(jax.random.key(0), 11)
    scale = CONFIG['weight_scale']
    k = CONFIG['kernel_size']
    c0 = CONFIG['in_channels']
    c1 = CONFIG['conv1_channels']
    c2 = CONFIG['conv2_channels']
    f0 = CONFIG['flat_features']
    f1 = CONFIG['fc1_features']
    f2 = CONFIG['fc2_features']
    f3 = CONFIG['fc3_features']

    x = jax.random.uniform(
        keys[0], (CONFIG['batch'], c0, CONFIG['height'], CONFIG['width']), dtype
    )
    conv1_w = jax.random.normal(keys[1], (c1, c0, k, k), dtype) * scale
    conv1_b = jax.random.normal(keys[2], (c1,), dtype) * scale
    conv2_w = jax.random.normal(keys[3], (c2, c1, k, k), dtype) * scale
    conv2_b = jax.random.normal(keys[4], (c2,), dtype) * scale
    fc1_w = jax.random.normal(keys[5], (f0, f1), dtype) * scale
    fc1_b = jax.random.normal(keys[6], (f1,), dtype) * scale
    fc2_w = jax.random.normal(keys[7], (f1, f2), dtype) * scale
    fc2_b = jax.random.normal(keys[8], (f2,), dtype) * scale
    fc3_w = jax.random.normal(keys[9], (f2, f3), dtype) * scale
    fc3_b = jax.random.normal(keys[10], (f3,), dtype) * scale
    return (
        x,
        conv1_w, conv1_b,
        conv2_w, conv2_b,
        fc1_w, fc1_b,
        fc2_w, fc2_b,
        fc3_w, fc3_b,
    )


def _conv_relu_pool(x, w, b):
    y = lax.conv_general_dilated(
        x,
        w,
        window_strides=(1, 1),
        padding='VALID',
        dimension_numbers=('NHWC', 'OIHW', 'NHWC'),
    )
    y = jax.nn.relu(y + b)
    window = (1, CONFIG['pool_size'], CONFIG['pool_size'], 1)
    strides = (1, CONFIG['pool_stride'], CONFIG['pool_stride'], 1)
    return lax.reduce_window(
        y, jnp.asarray(-jnp.inf, dtype=y.dtype), lax.max, window, strides, 'VALID'
    )


def workload(x, conv1_w, conv1_b, conv2_w, conv2_b,
             fc1_w, fc1_b, fc2_w, fc2_b, fc3_w, fc3_b):
    y = jnp.transpose(x, (0, 2, 3, 1))
    y = _conv_relu_pool(y, conv1_w, conv1_b)
    y = _conv_relu_pool(y, conv2_w, conv2_b)
    y = jnp.transpose(y, (0, 3, 1, 2))
    y = jnp.reshape(y, (-1, CONFIG['flat_features']))
    y = jax.nn.relu(y @ fc1_w + fc1_b)
    y = jax.nn.relu(y @ fc2_w + fc2_b)
    return y @ fc3_w + fc3_b
