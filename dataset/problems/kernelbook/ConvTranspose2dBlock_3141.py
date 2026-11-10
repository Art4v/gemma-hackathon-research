"""[bf16 variant] ConvTranspose2d (NCHW, PyTorch IOHW weight) -> per-channel bias -> ReLU."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'ConvTranspose2dBlock_3141',
    'batch': 8,
    'in_channels': 128,
    'out_channels': 128,
    'height': 256,
    'width': 256,
    'kernel_size': 4,
    'stride': 1,
    'padding': 0,
    'output_padding': 0,
}


def create_inputs(dtype=jnp.bfloat16):
    k_x, k_w, k_b = jax.random.split(jax.random.key(0), 3)
    x = jax.random.uniform(
        k_x,
        (CONFIG['batch'], CONFIG['in_channels'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    # PyTorch ConvTranspose2d weight layout: (in_channels, out_channels, kH, kW).
    weight = jax.random.normal(
        k_w,
        (CONFIG['in_channels'], CONFIG['out_channels'],
         CONFIG['kernel_size'], CONFIG['kernel_size']),
        dtype,
    ) * 0.02
    bias = jax.random.normal(k_b, (CONFIG['out_channels'],), dtype) * 0.02
    return x, weight, bias


def workload(x, weight, bias):
    stride = CONFIG['stride']
    padding = CONFIG['padding']
    output_padding = CONFIG['output_padding']
    k_h, k_w = weight.shape[2], weight.shape[3]

    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    y = lax.conv_general_dilated(
        x_nhwc,
        jnp.flip(weight, axis=(2, 3)),
        window_strides=(1, 1),
        padding=(
            (k_h - 1 - padding, k_h - 1 - padding + output_padding),
            (k_w - 1 - padding, k_w - 1 - padding + output_padding),
        ),
        lhs_dilation=(stride, stride),
        rhs_dilation=(1, 1),
        dimension_numbers=('NHWC', 'IOHW', 'NHWC'),
    )
    y = y + bias.reshape((1, 1, 1, -1))
    y = jnp.maximum(y, 0)
    return jnp.transpose(y, (0, 3, 1, 2))
