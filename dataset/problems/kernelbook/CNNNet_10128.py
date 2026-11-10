"""[bf16 variant] CNNNet: conv5-relu-maxpool2 x2, flatten, fc-relu-fc-relu-fc."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'CNNNet_10128',
    'batch': 64,
    'in_channels': 3,
    'height': 128,
    'width': 128,
    'conv1_features': 128,
    'conv2_features': 128,
    'kernel_size': 5,
    'conv_stride': 1,
    'conv_padding': 'VALID',
    'pool_size': 2,
    'pool_stride': 2,
    'pool_padding': 'VALID',
    'fc1_in': 107648,
    'fc1_out': 1024,
    'fc2_out': 128,
    'fc3_out': 128,
}


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG
    keys = jax.random.split(jax.random.key(0), 11)
    k = c['kernel_size']
    x = jax.random.uniform(
        keys[0], (c['batch'], c['in_channels'], c['height'], c['width']), dtype)
    conv1_w = jax.random.normal(
        keys[1], (c['conv1_features'], c['in_channels'], k, k), dtype) * 0.02
    conv1_b = jax.random.normal(keys[2], (c['conv1_features'],), dtype) * 0.02
    conv2_w = jax.random.normal(
        keys[3], (c['conv2_features'], c['conv1_features'], k, k), dtype) * 0.02
    conv2_b = jax.random.normal(keys[4], (c['conv2_features'],), dtype) * 0.02
    fc1_w = jax.random.normal(keys[5], (c['fc1_in'], c['fc1_out']), dtype) * 0.02
    fc1_b = jax.random.normal(keys[6], (c['fc1_out'],), dtype) * 0.02
    fc2_w = jax.random.normal(keys[7], (c['fc1_out'], c['fc2_out']), dtype) * 0.02
    fc2_b = jax.random.normal(keys[8], (c['fc2_out'],), dtype) * 0.02
    fc3_w = jax.random.normal(keys[9], (c['fc2_out'], c['fc3_out']), dtype) * 0.02
    fc3_b = jax.random.normal(keys[10], (c['fc3_out'],), dtype) * 0.02
    return (x, conv1_w, conv1_b, conv2_w, conv2_b,
            fc1_w, fc1_b, fc2_w, fc2_b, fc3_w, fc3_b)


def workload(x, conv1_w, conv1_b, conv2_w, conv2_b,
             fc1_w, fc1_b, fc2_w, fc2_b, fc3_w, fc3_b):
    c = CONFIG
    stride = (c['conv_stride'], c['conv_stride'])
    window = (1, c['pool_size'], c['pool_size'], 1)
    pool_strides = (1, c['pool_stride'], c['pool_stride'], 1)

    def conv(y, w, b):
        y = lax.conv_general_dilated(
            y, w, window_strides=stride, padding=c['conv_padding'],
            dimension_numbers=('NHWC', 'OIHW', 'NHWC'))
        return y + b

    def pool(y):
        return lax.reduce_window(
            y, -jnp.inf, lax.max, window, pool_strides, c['pool_padding'])

    y = jnp.transpose(x, (0, 2, 3, 1))
    y = pool(jax.nn.relu(conv(y, conv1_w, conv1_b)))
    y = pool(jax.nn.relu(conv(y, conv2_w, conv2_b)))
    y = jnp.transpose(y, (0, 3, 1, 2))
    y = jnp.reshape(y, (-1, fc1_w.shape[0]))
    y = jax.nn.relu(y @ fc1_w + fc1_b)
    y = jax.nn.relu(y @ fc2_w + fc2_b)
    return y @ fc3_w + fc3_b
