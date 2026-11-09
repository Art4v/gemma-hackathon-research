"""[bf16 variant] Conv2d (NCHW, k=11, stride=4, pad=2) + bias."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '50_conv_standard_2D__square_input__square_kernel',
    'batch': 256,
    'in_channels': 3,
    'out_channels': 96,
    'height': 224,
    'width': 224,
    'kernel_size': 11,
    'stride': 4,
    'padding': 2,
}


def create_inputs(dtype=jnp.bfloat16):
    k0, k1, k2 = jax.random.split(jax.random.key(0), 3)
    c = CONFIG
    x = jax.random.uniform(
        k0, (c['batch'], c['in_channels'], c['height'], c['width']), dtype)
    # Weight in HWIO layout: (kh, kw, in, out)
    w = jax.random.normal(
        k1, (c['kernel_size'], c['kernel_size'], c['in_channels'], c['out_channels']),
        dtype) * 0.02
    b = jax.random.normal(k2, (c['out_channels'],), dtype) * 0.02
    return (x, w, b)


def workload(x, w, b):
    stride = CONFIG['stride']
    pad = CONFIG['padding']
    # NCHW -> NHWC
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    y = lax.conv_general_dilated(
        x_nhwc, w,
        window_strides=(stride, stride),
        padding=((pad, pad), (pad, pad)),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
    )
    y = y + b.reshape(1, 1, 1, -1)
    # NHWC -> NCHW
    return jnp.transpose(y, (0, 3, 1, 2))
