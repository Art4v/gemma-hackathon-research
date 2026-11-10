"""ConvTranspose2d with caller-supplied filters and no bias, followed by ReLU."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'SimpleConvTranspose2dModule_14644',
    'batch': 8,
    'in_channels': 128,
    'out_channels_per_group': 128,
    'height': 256,
    'width': 256,
    'kernel_height': 4,
    'kernel_width': 4,
    'stride': 1,
    'padding': 0,
    'output_padding': 0,
    'dilation': 1,
    'groups': 1,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    k_inputs, k_filters = jax.random.split(key)
    inputs = jax.random.uniform(
        k_inputs,
        (CONFIG['batch'], CONFIG['in_channels'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    filters = jax.random.normal(
        k_filters,
        (
            CONFIG['in_channels'],
            CONFIG['out_channels_per_group'],
            CONFIG['kernel_height'],
            CONFIG['kernel_width'],
        ),
        dtype,
    ) * 0.02
    return inputs, filters


def workload(inputs, filters):
    stride = CONFIG['stride']
    padding = CONFIG['padding']
    output_padding = CONFIG['output_padding']
    dilation = CONFIG['dilation']
    groups = CONFIG['groups']

    in_channels, out_per_group, kernel_h, kernel_w = filters.shape
    in_per_group = in_channels // groups

    # PyTorch (in, out/groups, H, W) -> lax (in/groups, out, H, W).
    regrouped = jnp.transpose(
        filters.reshape(groups, in_per_group, out_per_group, kernel_h, kernel_w),
        (1, 0, 2, 3, 4),
    ).reshape(in_per_group, groups * out_per_group, kernel_h, kernel_w)
    # lax convolution reverses the kernel; a transposed convolution does not.
    kernel = jnp.flip(jnp.transpose(regrouped, (2, 3, 0, 1)), axis=(0, 1))

    pad_h = dilation * (kernel_h - 1) - padding
    pad_w = dilation * (kernel_w - 1) - padding
    outputs = lax.conv_general_dilated(
        jnp.transpose(inputs, (0, 2, 3, 1)),
        kernel,
        window_strides=(1, 1),
        padding=((pad_h, pad_h + output_padding), (pad_w, pad_w + output_padding)),
        lhs_dilation=(stride, stride),
        rhs_dilation=(dilation, dilation),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        feature_group_count=groups,
    )
    return jnp.transpose(jnp.maximum(outputs, 0), (0, 3, 1, 2))
