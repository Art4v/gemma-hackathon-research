"""[bf16 variant] WSConv2d: equalized-LR input scaling, Conv2d (no internal bias), per-channel bias addition."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'WSConv2d_3729',
    'batch': 8,
    'in_channels': 128,
    'out_channels': 128,
    'height': 256,
    'width': 256,
    'kernel_size': 3,
    'stride': 1,
    'padding': 1,
    'gain': 2,
    'weight_std': 0.02,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k_x, k_w, k_b = jax.random.split(key, 3)
    x = jax.random.uniform(
        k_x,
        (CONFIG['batch'], CONFIG['in_channels'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    weight = jax.random.normal(
        k_w,
        (CONFIG['out_channels'], CONFIG['in_channels'], CONFIG['kernel_size'], CONFIG['kernel_size']),
        dtype,
    ) * CONFIG['weight_std']
    bias = jax.random.normal(k_b, (CONFIG['out_channels'],), dtype) * CONFIG['weight_std']
    return x, weight, bias


def workload(x, weight, bias):
    out_channels, in_channels, kernel_h, kernel_w = weight.shape
    scale = (CONFIG['gain'] / (in_channels * kernel_h * kernel_w)) ** 0.5
    stride = CONFIG['stride']
    padding = CONFIG['padding']
    y = jnp.transpose(x * scale, (0, 2, 3, 1))
    y = lax.conv_general_dilated(
        y,
        weight,
        window_strides=(stride, stride),
        padding=((padding, padding), (padding, padding)),
        dimension_numbers=('NHWC', 'OIHW', 'NHWC'),
    )
    y = jnp.transpose(y, (0, 3, 1, 2))
    return y + jnp.reshape(bias, (1, out_channels, 1, 1))
