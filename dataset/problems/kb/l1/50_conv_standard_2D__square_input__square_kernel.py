"""2D standard convolution with square input and square kernel."""

import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '50_conv_standard_2D__square_input__square_kernel',
    'batch_size': 512,
    'in_channels': 3,
    'out_channels': 96,
    'height': 256,
    'width': 256,
    'kernel_size': 11,
    'stride': 4,
    'padding': 2,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    k1, k2, k3 = jax.random.split(key, 3)
    B = CONFIG['batch_size']
    C_in = CONFIG['in_channels']
    C_out = CONFIG['out_channels']
    H = CONFIG['height']
    W = CONFIG['width']
    K = CONFIG['kernel_size']

    x = jax.random.uniform(k1, (B, C_in, H, W), dtype=dtype)
    weight = jax.random.normal(k2, (C_out, C_in, K, K), dtype=dtype) * 0.02
    bias = jax.random.normal(k3, (C_out,), dtype=dtype) * 0.02
    return x, weight, bias


def workload(x, weight, bias):
    stride = CONFIG['stride']
    pad = CONFIG['padding']

    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    w_hwio = jnp.transpose(weight, (2, 3, 1, 0))

    out_nhwc = lax.conv_general_dilated(
        x_nhwc,
        w_hwio,
        window_strides=(stride, stride),
        padding=((pad, pad), (pad, pad)),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
    )
    out_nhwc = out_nhwc + bias.reshape(1, 1, 1, -1)
    out_nchw = jnp.transpose(out_nhwc, (0, 3, 1, 2))
    return out_nchw
