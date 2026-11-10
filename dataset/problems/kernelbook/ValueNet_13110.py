"""ValueNet: conv-relu x3, 2x2 max pool, flatten, linear-relu, linear."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'ValueNet_13110',
    'batch': 16,
    'in_channels': 4,
    'height': 512,
    'width': 512,
    'conv1_out': 32,
    'conv1_kernel': 8,
    'conv1_stride': 4,
    'conv1_padding': 2,
    'conv2_out': 64,
    'conv2_kernel': 4,
    'conv2_stride': 2,
    'conv2_padding': 1,
    'conv3_out': 64,
    'conv3_kernel': 3,
    'conv3_stride': 1,
    'conv3_padding': 1,
    'pool_size': 2,
    'fc1_in': 65536,
    'fc1_out': 512,
    'actions': 128,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    k = jax.random.split(jax.random.key(0), 11)
    x = jax.random.uniform(
        k[0], (c['batch'], c['in_channels'], c['height'], c['width']), dtype)
    conv1_w = jax.random.normal(
        k[1], (c['conv1_kernel'], c['conv1_kernel'], c['in_channels'], c['conv1_out']), dtype) * 0.02
    conv1_b = jax.random.normal(k[2], (c['conv1_out'],), dtype) * 0.02
    conv2_w = jax.random.normal(
        k[3], (c['conv2_kernel'], c['conv2_kernel'], c['conv1_out'], c['conv2_out']), dtype) * 0.02
    conv2_b = jax.random.normal(k[4], (c['conv2_out'],), dtype) * 0.02
    conv3_w = jax.random.normal(
        k[5], (c['conv3_kernel'], c['conv3_kernel'], c['conv2_out'], c['conv3_out']), dtype) * 0.02
    conv3_b = jax.random.normal(k[6], (c['conv3_out'],), dtype) * 0.02
    fc1_w = jax.random.normal(k[7], (c['fc1_in'], c['fc1_out']), dtype) * 0.02
    fc1_b = jax.random.normal(k[8], (c['fc1_out'],), dtype) * 0.02
    fc2_w = jax.random.normal(k[9], (c['fc1_out'], c['actions']), dtype) * 0.02
    fc2_b = jax.random.normal(k[10], (c['actions'],), dtype) * 0.02
    return (x, conv1_w, conv1_b, conv2_w, conv2_b, conv3_w, conv3_b,
            fc1_w, fc1_b, fc2_w, fc2_b)


def _conv_relu(x, w, b, stride, pad):
    y = lax.conv_general_dilated(
        x, w,
        window_strides=(stride, stride),
        padding=((pad, pad), (pad, pad)),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
    )
    return jnp.maximum(y + b, 0)


def workload(x, conv1_w, conv1_b, conv2_w, conv2_b, conv3_w, conv3_b,
             fc1_w, fc1_b, fc2_w, fc2_b):
    c = CONFIG
    y = jnp.transpose(x, (0, 2, 3, 1))
    y = _conv_relu(y, conv1_w, conv1_b, c['conv1_stride'], c['conv1_padding'])
    y = _conv_relu(y, conv2_w, conv2_b, c['conv2_stride'], c['conv2_padding'])
    y = _conv_relu(y, conv3_w, conv3_b, c['conv3_stride'], c['conv3_padding'])
    p = c['pool_size']
    y = lax.reduce_window(y, -jnp.inf, lax.max, (1, p, p, 1), (1, p, p, 1), 'VALID')
    y = jnp.transpose(y, (0, 3, 1, 2))
    y = y.reshape(-1, fc1_w.shape[0])
    y = jnp.maximum(y @ fc1_w + fc1_b, 0)
    return y @ fc2_w + fc2_b
