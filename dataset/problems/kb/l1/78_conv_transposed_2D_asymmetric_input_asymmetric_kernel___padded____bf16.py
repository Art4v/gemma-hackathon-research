"""[bf16 variant] 2D transposed convolution with asymmetric input, asymmetric kernel, and padding."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '78_conv_transposed_2D_asymmetric_input_asymmetric_kernel___padded__',
    'batch_size': 8,
    'in_channels': 32,
    'out_channels': 32,
    'height': 512,
    'width': 1024,
    'kernel_h': 3,
    'kernel_w': 7,
    'stride_h': 1,
    'stride_w': 1,
    'padding_h': 1,
    'padding_w': 3,
}


def create_inputs(dtype=jnp.bfloat16):
    k1, k2 = jax.random.split(jax.random.key(0), 2)
    x = jax.random.uniform(
        k1,
        (CONFIG['batch_size'], CONFIG['in_channels'], CONFIG['height'], CONFIG['width']),
        dtype=dtype,
    )
    weight = (
        jax.random.normal(
            k2,
            (CONFIG['in_channels'], CONFIG['out_channels'], CONFIG['kernel_h'], CONFIG['kernel_w']),
            dtype=dtype,
        )
        * 0.02
    )
    return (x, weight)


def workload(x, weight):
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    w_hwio = jnp.transpose(weight, (2, 3, 0, 1))
    w_flipped = jnp.flip(w_hwio, axis=(0, 1))

    kh, kw = weight.shape[2], weight.shape[3]
    pad_h = kh - 1 - CONFIG['padding_h']
    pad_w = kw - 1 - CONFIG['padding_w']

    out_nhwc = lax.conv_general_dilated(
        lhs=x_nhwc,
        rhs=w_flipped,
        window_strides=(1, 1),
        padding=((pad_h, pad_h), (pad_w, pad_w)),
        lhs_dilation=(CONFIG['stride_h'], CONFIG['stride_w']),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
    )
    return jnp.transpose(out_nhwc, (0, 3, 1, 2))
