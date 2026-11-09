"""[bf16 variant] ConvTranspose2d followed by scalar multiplication and two global average poolings."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '44_ConvTranspose2d_Multiply_GlobalAvgPool_GlobalAvgPool_Mean',
    'batch_size': 32,
    'in_channels': 128,
    'out_channels': 128,
    'height': 128,
    'width': 128,
    'kernel_size': 3,
    'stride': 2,
    'padding': 1,
    'output_padding': 1,
    'multiplier': 0.5,
}

def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k1, k2, k3 = jax.random.split(key, 3)
    x = jax.random.uniform(
        k1,
        (CONFIG['batch_size'], CONFIG['in_channels'], CONFIG['height'], CONFIG['width']),
        dtype=dtype,
    )
    weight = jax.random.normal(
        k2,
        (CONFIG['in_channels'], CONFIG['out_channels'], CONFIG['kernel_size'], CONFIG['kernel_size']),
        dtype=dtype,
    ) * 0.02
    bias = jax.random.normal(
        k3,
        (CONFIG['out_channels'],),
        dtype=dtype,
    ) * 0.02
    return (x, weight, bias)

def workload(x, weight, bias):
    stride = CONFIG['stride']
    padding = CONFIG['padding']
    output_padding = CONFIG['output_padding']
    multiplier = CONFIG['multiplier']
    kh, kw = weight.shape[2], weight.shape[3]

    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    w_hwio = jnp.transpose(weight, (2, 3, 1, 0))

    pad_h = (kh - 1 - padding, kh - 1 - padding + output_padding)
    pad_w = (kw - 1 - padding, kw - 1 - padding + output_padding)

    out_nhwc = lax.conv_transpose(
        x_nhwc,
        w_hwio,
        strides=(stride, stride),
        padding=(pad_h, pad_w),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        transpose_kernel=True,
    )
    out_nhwc = out_nhwc + bias.reshape(1, 1, 1, -1)
    out = jnp.transpose(out_nhwc, (0, 3, 1, 2))

    out = out * multiplier
    out = jnp.mean(out, axis=(2, 3), keepdims=True)
    out = jnp.mean(out, axis=(2, 3), keepdims=True)
    return out
