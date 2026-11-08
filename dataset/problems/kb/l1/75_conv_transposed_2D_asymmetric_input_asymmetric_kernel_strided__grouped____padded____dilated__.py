"""2D transposed convolution with asymmetric input, asymmetric kernel, stride, padding, dilation, and groups."""

import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '75_conv_transposed_2D_asymmetric_input_asymmetric_kernel_strided__grouped____padded____dilated__',
    'batch_size': 32,
    'in_channels': 128,
    'out_channels': 128,
    'height': 128,
    'width': 256,
    'kernel_size': (3, 5),
    'stride': (2, 3),
    'padding': (1, 2),
    'dilation': (2, 1),
    'groups': 4,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    k1, k2 = jax.random.split(key, 2)
    b = CONFIG['batch_size']
    c_in = CONFIG['in_channels']
    c_out = CONFIG['out_channels']
    h = CONFIG['height']
    w = CONFIG['width']
    kh, kw = CONFIG['kernel_size']
    g = CONFIG['groups']

    x = jax.random.uniform(k1, (b, c_in, h, w), dtype=dtype)
    weight = jax.random.normal(k2, (c_in, c_out // g, kh, kw), dtype=dtype) * 0.02
    return (x, weight)


def workload(x, weight):
    stride = CONFIG['stride']
    padding = CONFIG['padding']
    dilation = CONFIG['dilation']
    groups = CONFIG['groups']

    c_in, c_out_per_g, kh, kw = weight.shape
    c_out = c_out_per_g * groups

    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))

    w = jnp.flip(weight, axis=(2, 3))
    w = jnp.reshape(w, (groups, c_in // groups, c_out_per_g, kh, kw))
    w = jnp.transpose(w, (0, 2, 1, 3, 4))
    w = jnp.reshape(w, (c_out, c_in // groups, kh, kw))
    w_hwio = jnp.transpose(w, (2, 3, 1, 0))

    eff_kh = (kh - 1) * dilation[0] + 1
    eff_kw = (kw - 1) * dilation[1] + 1
    pad_h = eff_kh - 1 - padding[0]
    pad_w = eff_kw - 1 - padding[1]
    jax_padding = ((pad_h, pad_h), (pad_w, pad_w))

    out_nhwc = lax.conv_general_dilated(
        lhs=x_nhwc,
        rhs=w_hwio,
        window_strides=(1, 1),
        padding=jax_padding,
        lhs_dilation=stride,
        rhs_dilation=dilation,
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        feature_group_count=groups,
    )

    return jnp.transpose(out_nhwc, (0, 3, 1, 2))
