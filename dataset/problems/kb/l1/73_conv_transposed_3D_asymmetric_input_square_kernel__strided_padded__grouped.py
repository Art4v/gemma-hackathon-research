"""3D grouped strided padded transposed convolution with asymmetric input and square kernel."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '73_conv_transposed_3D_asymmetric_input_square_kernel__strided_padded__grouped',
    'batch_size': 8,
    'in_channels': 128,
    'out_channels': 128,
    'kernel_size': 3,
    'depth': 16,
    'height': 32,
    'width': 64,
    'stride': 2,
    'padding': 1,
    'groups': 4,
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
    weight = (
        jax.random.normal(
            k2,
            (
                CONFIG['in_channels'],
                CONFIG['out_channels'] // CONFIG['groups'],
                CONFIG['kernel_size'],
                CONFIG['kernel_size'],
                CONFIG['kernel_size'],
            ),
            dtype=dtype,
        )
        * 0.02
    )
    return (x, weight)


def workload(x, weight):
    in_channels = x.shape[1]
    out_channels_per_group = weight.shape[1]
    groups = in_channels // out_channels_per_group
    kd, kh, kw = weight.shape[2], weight.shape[3], weight.shape[4]
    stride = CONFIG['stride']
    padding = CONFIG['padding']

    x_ndhwc = jnp.transpose(x, (0, 2, 3, 4, 1))

    w = jnp.reshape(
        weight,
        (groups, in_channels // groups, out_channels_per_group, kd, kh, kw),
    )
    w = jnp.flip(w, axis=(3, 4, 5))
    w = jnp.transpose(w, (3, 4, 5, 1, 0, 2))
    w_dhwio = jnp.reshape(
        w, (kd, kh, kw, in_channels // groups, groups * out_channels_per_group)
    )

    pad_d = kd - 1 - padding
    pad_h = kh - 1 - padding
    pad_w = kw - 1 - padding

    out_ndhwc = lax.conv_general_dilated(
        lhs=x_ndhwc,
        rhs=w_dhwio,
        window_strides=(1, 1, 1),
        padding=((pad_d, pad_d), (pad_h, pad_h), (pad_w, pad_w)),
        lhs_dilation=(stride, stride, stride),
        dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC'),
        feature_group_count=groups,
    )
    return jnp.transpose(out_ndhwc, (0, 4, 1, 2, 3))
