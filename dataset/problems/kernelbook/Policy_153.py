"""[bf16 variant] Policy_153: conv2d (no bias) -> relu -> conv2d (bias) -> relu -> NCHW flatten -> linear -> relu -> linear -> sigmoid."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'Policy_153',
    'batch': 64,
    'in_channels': 2,
    'image_size': 512,
    'conv1_channels': 4,
    'conv1_kernel': 6,
    'conv1_stride': 2,
    'conv2_channels': 16,
    'conv2_kernel': 6,
    'conv2_stride': 4,
    'conv2_output_size': 63,
    'fc_in': 63504,
    'fc_hidden': 256,
    'fc_out': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    keys = jax.random.split(jax.random.key(0), 8)
    c = CONFIG
    x = jax.random.uniform(
        keys[0],
        (c['batch'], c['in_channels'], c['image_size'], c['image_size']),
        dtype,
    )
    conv1_w = jax.random.normal(
        keys[1],
        (c['conv1_channels'], c['in_channels'], c['conv1_kernel'], c['conv1_kernel']),
        dtype,
    ) * 0.02
    conv2_w = jax.random.normal(
        keys[2],
        (c['conv2_channels'], c['conv1_channels'], c['conv2_kernel'], c['conv2_kernel']),
        dtype,
    ) * 0.02
    conv2_b = jax.random.normal(keys[3], (c['conv2_channels'],), dtype) * 0.02
    fc1_w = jax.random.normal(keys[4], (c['fc_in'], c['fc_hidden']), dtype) * 0.02
    fc1_b = jax.random.normal(keys[5], (c['fc_hidden'],), dtype) * 0.02
    fc2_w = jax.random.normal(keys[6], (c['fc_hidden'], c['fc_out']), dtype) * 0.02
    fc2_b = jax.random.normal(keys[7], (c['fc_out'],), dtype) * 0.02
    return x, conv1_w, conv2_w, conv2_b, fc1_w, fc1_b, fc2_w, fc2_b


def _conv(x, w, stride):
    return lax.conv_general_dilated(
        x,
        w,
        window_strides=(stride, stride),
        padding='VALID',
        dimension_numbers=('NHWC', 'OIHW', 'NHWC'),
    )


def workload(x, conv1_w, conv2_w, conv2_b, fc1_w, fc1_b, fc2_w, fc2_b):
    h = jnp.transpose(x, (0, 2, 3, 1))
    h = jax.nn.relu(_conv(h, conv1_w, CONFIG['conv1_stride']))
    h = jax.nn.relu(_conv(h, conv2_w, CONFIG['conv2_stride']) + conv2_b)
    h = jnp.transpose(h, (0, 3, 1, 2))
    h = jnp.reshape(h, (-1, fc1_w.shape[0]))
    h = jax.nn.relu(h @ fc1_w + fc1_b)
    return jax.nn.sigmoid(h @ fc2_w + fc2_b)
