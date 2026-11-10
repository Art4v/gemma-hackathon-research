"""BasicResidualBlock: Conv2d -> PReLU -> Conv2d -> input skip add -> PReLU."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'BasicResidualBlock_10970',
    'batch': 8,
    'in_channels': 128,
    'channels': 128,
    'height': 256,
    'width': 256,
    'kernel_size': 3,
    'conv1_stride': 1,
    'conv2_stride': 1,
    'padding': 1,
    'prelu_num_parameters': 1,
    'prelu_init': 0.25,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.float32):
    keys = jax.random.split(jax.random.key(0), 5)
    k = CONFIG['kernel_size']
    in_c = CONFIG['in_channels']
    c = CONFIG['channels']
    scale = CONFIG['weight_scale']
    prelu_shape = (CONFIG['prelu_num_parameters'],)

    x = jax.random.uniform(
        keys[0],
        (CONFIG['batch'], in_c, CONFIG['height'], CONFIG['width']),
        dtype,
    )
    w1 = jax.random.normal(keys[1], (c, in_c, k, k), dtype) * scale
    b1 = jax.random.normal(keys[2], (c,), dtype) * scale
    a1 = jnp.full(prelu_shape, CONFIG['prelu_init'], dtype)
    w2 = jax.random.normal(keys[3], (c, c, k, k), dtype) * scale
    b2 = jax.random.normal(keys[4], (c,), dtype) * scale
    a2 = jnp.full(prelu_shape, CONFIG['prelu_init'], dtype)
    return x, w1, b1, a1, w2, b2, a2


def _conv(x, w, b, stride):
    p = CONFIG['padding']
    y = lax.conv_general_dilated(
        x,
        w,
        window_strides=(stride, stride),
        padding=((p, p), (p, p)),
        dimension_numbers=('NHWC', 'OIHW', 'NHWC'),
    )
    return y + b


def _prelu(x, a):
    return jnp.where(x >= 0, x, a * x)


def workload(x, w1, b1, a1, w2, b2, a2):
    h = jnp.transpose(x, (0, 2, 3, 1))
    out = _prelu(_conv(h, w1, b1, CONFIG['conv1_stride']), a1)
    out = _conv(out, w2, b2, CONFIG['conv2_stride'])
    out = out + h
    out = _prelu(out, a2)
    return jnp.transpose(out, (0, 3, 1, 2))
