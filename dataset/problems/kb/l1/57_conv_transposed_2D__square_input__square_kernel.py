"""Transposed 2D convolution with square input and square kernel."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '57_conv_transposed_2D__square_input__square_kernel',
    'batch_size': 8,
    'in_channels': 64,
    'out_channels': 64,
    'kernel_size': 3,
    'height': 256,
    'width': 256,
    'stride': 1,
    'padding': 0,
    'output_padding': 0,
    'groups': 1,
}


def create_inputs(dtype=jnp.float32):
    k1, k2 = jax.random.split(jax.random.key(0), 2)
    x = jax.random.uniform(
        k1,
        (CONFIG['batch_size'], CONFIG['in_channels'], CONFIG['height'], CONFIG['width']),
        dtype=dtype,
    )
    weight = (
        jax.random.normal(
            k2,
            (
                CONFIG['in_channels'],
                CONFIG['out_channels'] // CONFIG['groups'],
                CONFIG['kernel_size'],
                CONFIG['kernel_size'],
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
    kh, kw = weight.shape[2], weight.shape[3]

    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    w_flipped = jnp.flip(weight, axis=(2, 3))
    if groups > 1:
        in_per_group = weight.shape[0] // groups
        out_per_group = weight.shape[1]
        w_flipped = jnp.reshape(w_flipped, (groups, in_per_group, out_per_group, kh, kw))
        w_flipped = jnp.transpose(w_flipped, (1, 0, 2, 3, 4))
        w_flipped = jnp.reshape(w_flipped, (in_per_group, groups * out_per_group, kh, kw))

    pad_h = (kh - 1 - padding, kh - 1 - padding + output_padding)
    pad_w = (kw - 1 - padding, kw - 1 - padding + output_padding)

    out_nhwc = lax.conv_general_dilated(
        lhs=x_nhwc,
        rhs=w_flipped,
        window_strides=(1, 1),
        padding=(pad_h, pad_w),
        lhs_dilation=(stride, stride),
        dimension_numbers=('NHWC', 'IOHW', 'NHWC'),
        feature_group_count=groups,
    )
    return jnp.transpose(out_nhwc, (0, 3, 1, 2))
