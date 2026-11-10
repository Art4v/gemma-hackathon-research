"""StridedNet_17466: dilated Conv2d + dilated MaxPool2d + exact GELU chain ending in 1x1 convs."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'StridedNet_17466',
    'batch': 8,
    'in_channels': 1,
    'height': 2048,
    'width': 2048,
    # (in_channels, out_channels, kernel_size, dilation) for conv1, conv2, fc1, fc2, fc3
    'conv_layers': (
        (1, 10, 6, 1),
        (10, 20, 4, 2),
        (20, 32, 2, 4),
        (32, 16, 1, 1),
        (16, 1, 1, 1),
    ),
    'conv_stride': 1,
    'pool_kernel': 2,
    'pool_stride': 1,
    'pool1_dilation': 1,
    'pool2_dilation': 2,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.float32):
    layers = CONFIG['conv_layers']
    keys = jax.random.split(jax.random.key(0), 1 + 2 * len(layers))
    x = jax.random.uniform(
        keys[0],
        (CONFIG['batch'], CONFIG['in_channels'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    params = []
    for i, (c_in, c_out, k, _) in enumerate(layers):
        w = jax.random.normal(keys[1 + 2 * i], (c_out, c_in, k, k), dtype) * CONFIG['weight_scale']
        b = jax.random.normal(keys[2 + 2 * i], (c_out,), dtype) * CONFIG['weight_scale']
        params.extend([w, b])
    return (x, *params)


def _conv(x, w, b, dilation):
    stride = CONFIG['conv_stride']
    y = lax.conv_general_dilated(
        x,
        w,
        window_strides=(stride, stride),
        padding='VALID',
        rhs_dilation=(dilation, dilation),
        dimension_numbers=('NHWC', 'OIHW', 'NHWC'),
    )
    return y + b


def _max_pool(x, dilation):
    k = CONFIG['pool_kernel']
    s = CONFIG['pool_stride']
    return lax.reduce_window(
        x,
        jnp.array(-jnp.inf, dtype=x.dtype),
        lax.max,
        window_dimensions=(1, k, k, 1),
        window_strides=(1, s, s, 1),
        padding='VALID',
        window_dilation=(1, dilation, dilation, 1),
    )


def _gelu(x):
    return jax.nn.gelu(x, approximate=False)


def workload(x, w1, b1, w2, b2, w3, b3, w4, b4, w5, b5):
    dilations = [layer[3] for layer in CONFIG['conv_layers']]
    out = jnp.transpose(x, (0, 2, 3, 1))
    out = _gelu(_max_pool(_conv(out, w1, b1, dilations[0]), CONFIG['pool1_dilation']))
    out = _gelu(_max_pool(_conv(out, w2, b2, dilations[1]), CONFIG['pool2_dilation']))
    out = _gelu(_conv(out, w3, b3, dilations[2]))
    out = _gelu(_conv(out, w4, b4, dilations[3]))
    out = _conv(out, w5, b5, dilations[4])
    return jnp.transpose(out, (0, 3, 1, 2))
