"""Transposed 2D convolution, bias subtraction, and tanh activation."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '5_ConvTranspose2d_Subtract_Tanh',
    'batch_size': 32,
    'in_channels': 64,
    'out_channels': 64,
    'height': 256,
    'width': 256,
    'kernel_size': 4,
    'stride': 2,
    'padding': 1,
    'output_padding': 1,
}

def create_inputs(dtype=jnp.float32):
    k1, k2, k3, k4 = jax.random.split(jax.random.key(0), 4)
    B = CONFIG['batch_size']
    C_in = CONFIG['in_channels']
    C_out = CONFIG['out_channels']
    H = CONFIG['height']
    W = CONFIG['width']
    K = CONFIG['kernel_size']
    x = jax.random.uniform(k1, (B, C_in, H, W), dtype=dtype)
    weight = jax.random.normal(k2, (C_in, C_out, K, K), dtype=dtype) * 0.02
    conv_bias = jax.random.normal(k3, (C_out,), dtype=dtype) * 0.02
    sub_bias = jax.random.normal(k4, (C_out, 1, 1), dtype=dtype) * 0.02
    return (x, weight, conv_bias, sub_bias)

def workload(x, weight, conv_bias, sub_bias):
    stride = CONFIG['stride']
    padding = CONFIG['padding']
    output_padding = CONFIG['output_padding']
    K = weight.shape[2]
    pad_low = K - 1 - padding
    pad_high = K - 1 - padding + output_padding
    
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    w_hwio = jnp.transpose(weight, (2, 3, 1, 0))
    
    out_nhwc = lax.conv_transpose(
        x_nhwc,
        w_hwio,
        strides=(stride, stride),
        padding=((pad_low, pad_high), (pad_low, pad_high)),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        transpose_kernel=True,
    )
    out = jnp.transpose(out_nhwc, (0, 3, 1, 2))
    out = out + jnp.reshape(conv_bias, (1, -1, 1, 1))
    out = out - sub_bias
    out = jnp.tanh(out)
    return out
