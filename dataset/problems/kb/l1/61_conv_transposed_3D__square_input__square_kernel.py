"""3D transposed convolution with cube input and cube kernel."""

import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '61_conv_transposed_3D__square_input__square_kernel',
    'batch_size': 8,
    'in_channels': 64,
    'out_channels': 64,
    'kernel_size': 3,
    'depth': 64,
    'height': 64,
    'width': 64,
    'stride': 1,
    'padding': 0,
    'output_padding': 0,
    'groups': 1,
}


def create_inputs(dtype=jnp.float32):
    k1, k2 = jax.random.split(jax.random.key(0), 2)
    x = jax.random.uniform(
        k1,
        (
            CONFIG['batch_size'],
            CONFIG['in_channels'],
            CONFIG['depth'],
            CONFIG['height'],
            CONFIG['width'],
        ),
        dtype=dtype,
    )
    k = CONFIG['kernel_size']
    weight = (
        jax.random.normal(
            k2,
            (
                CONFIG['in_channels'],
                CONFIG['out_channels'] // CONFIG['groups'],
                k,
                k,
                k,
            ),
            dtype=dtype,
        )
        * 0.02
    )
    return (x, weight)


def workload(x, weight):
    stride = CONFIG['stride']
    padding = CONFIG['padding']
    output_padding = CONFIG['output_padding']
    groups = CONFIG['groups']

    in_c = weight.shape[0]
    out_c_per_g = weight.shape[1]
    kd, kh, kw = weight.shape[2], weight.shape[3], weight.shape[4]

    x_ndhwc = jnp.transpose(x, (0, 2, 3, 4, 1))

    w_flipped = jnp.flip(weight, axis=(2, 3, 4))
    w_grouped = w_flipped.reshape(groups, in_c // groups, out_c_per_g, kd, kh, kw)
    w_grouped = jnp.transpose(w_grouped, (1, 0, 2, 3, 4, 5))
    w_iodhw = w_grouped.reshape(in_c // groups, groups * out_c_per_g, kd, kh, kw)

    pad_d = (kd - 1 - padding, kd - 1 - padding + output_padding)
    pad_h = (kh - 1 - padding, kh - 1 - padding + output_padding)
    pad_w = (kw - 1 - padding, kw - 1 - padding + output_padding)

    out_ndhwc = lax.conv_general_dilated(
        lhs=x_ndhwc,
        rhs=w_iodhw,
        window_strides=(1, 1, 1),
        padding=(pad_d, pad_h, pad_w),
        lhs_dilation=(stride, stride, stride),
        dimension_numbers=('NDHWC', 'IODHW', 'NDHWC'),
        feature_group_count=groups,
    )
    return jnp.transpose(out_ndhwc, (0, 4, 1, 2, 3))
