"""2D transposed convolution with asymmetric input, square kernel, dilation, padding, and stride."""

import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '81_conv_transposed_2D_asymmetric_input_square_kernel___dilated____padded____strided__',
    'batch_size': 32,
    'in_channels': 128,
    'out_channels': 128,
    'kernel_size': 3,
    'height_in': 64,
    'width_in': 128,
    'stride': 5,
    'padding': 1,
    'dilation': 2,
}


def create_inputs(dtype=jnp.float32):
    k1, k2 = jax.random.split(jax.random.key(0), 2)
    x = jax.random.uniform(
        k1,
        (CONFIG['batch_size'], CONFIG['in_channels'], CONFIG['height_in'], CONFIG['width_in']),
        dtype=dtype,
    )
    weight = (
        jax.random.normal(
            k2,
            (CONFIG['in_channels'], CONFIG['out_channels'], CONFIG['kernel_size'], CONFIG['kernel_size']),
            dtype=dtype,
        )
        * 0.02
    )
    return (x, weight)


def workload(x, weight):
    stride = CONFIG['stride']
    padding = CONFIG['padding']
    dilation = CONFIG['dilation']
    kh, kw = weight.shape[2], weight.shape[3]

    pad_h = (kh - 1) * dilation - padding
    pad_w = (kw - 1) * dilation - padding

    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    w_flipped = jnp.flip(weight, axis=(2, 3))
    w_hwio = jnp.transpose(w_flipped, (2, 3, 0, 1))

    out_nhwc = lax.conv_general_dilated(
        lhs=x_nhwc,
        rhs=w_hwio,
        window_strides=(1, 1),
        padding=((pad_h, pad_h), (pad_w, pad_w)),
        lhs_dilation=(stride, stride),
        rhs_dilation=(dilation, dilation),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
    )
    return jnp.transpose(out_nhwc, (0, 3, 1, 2))
