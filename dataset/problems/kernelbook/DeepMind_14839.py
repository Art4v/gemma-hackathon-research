"""[bf16 variant] DeepMind_14839: (Conv2d VALID + ReLU) x3 -> NCHW flatten to fc_in rows -> Linear + ReLU."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'DeepMind_14839',
    'batch': 8,
    'in_channels': 4,
    'height': 1152,
    'width': 1152,
    'conv1_channels': 32,
    'conv2_channels': 64,
    'conv3_channels': 32,
    'conv1_kernel': 8,
    'conv2_kernel': 4,
    'conv3_kernel': 3,
    'conv1_stride': 4,
    'conv2_stride': 2,
    'conv3_stride': 1,
    'fc_in': 32 * 7 * 7,
    'fc_out': 512,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG
    key = jax.random.key(0)
    kx, k1, k2, k3, kf = jax.random.split(key, 5)
    x = jax.random.uniform(
        kx, (c['batch'], c['in_channels'], c['height'], c['width']), dtype)
    w1 = jax.random.normal(
        k1,
        (c['conv1_channels'], c['in_channels'], c['conv1_kernel'], c['conv1_kernel']),
        dtype) * c['weight_scale']
    b1 = jnp.zeros((c['conv1_channels'],), dtype)
    w2 = jax.random.normal(
        k2,
        (c['conv2_channels'], c['conv1_channels'], c['conv2_kernel'], c['conv2_kernel']),
        dtype) * c['weight_scale']
    b2 = jnp.zeros((c['conv2_channels'],), dtype)
    w3 = jax.random.normal(
        k3,
        (c['conv3_channels'], c['conv2_channels'], c['conv3_kernel'], c['conv3_kernel']),
        dtype) * c['weight_scale']
    b3 = jnp.zeros((c['conv3_channels'],), dtype)
    fw = jax.random.normal(kf, (c['fc_in'], c['fc_out']), dtype) * c['weight_scale']
    fb = jnp.zeros((c['fc_out'],), dtype)
    return x, w1, b1, w2, b2, w3, b3, fw, fb


def _conv_relu(x, w, b, stride):
    y = lax.conv_general_dilated(
        x,
        w,
        window_strides=(stride, stride),
        padding='VALID',
        dimension_numbers=('NHWC', 'OIHW', 'NHWC'),
    )
    return jax.nn.relu(y + b)


def workload(x, w1, b1, w2, b2, w3, b3, fw, fb):
    c = CONFIG
    y = jnp.transpose(x, (0, 2, 3, 1))
    y = _conv_relu(y, w1, b1, c['conv1_stride'])
    y = _conv_relu(y, w2, b2, c['conv2_stride'])
    y = _conv_relu(y, w3, b3, c['conv3_stride'])
    y = jnp.transpose(y, (0, 3, 1, 2))
    y = jnp.reshape(y, (-1, c['fc_in']))
    return jax.nn.relu(y @ fw + fb)
