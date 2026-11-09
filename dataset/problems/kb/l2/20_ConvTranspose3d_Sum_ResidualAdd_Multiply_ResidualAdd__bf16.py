"""[bf16 variant] 3D transposed convolution followed by bias add, residual add, multiply, and residual add."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '20_ConvTranspose3d_Sum_ResidualAdd_Multiply_ResidualAdd',
    'batch_size': 16,
    'in_channels': 64,
    'out_channels': 64,
    'depth': 32,
    'height': 32,
    'width': 32,
    'kernel_size': 3,
    'stride': 2,
    'padding': 1,
    'output_padding': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    k0, k1, k2 = jax.random.split(jax.random.key(0), 3)
    B = CONFIG['batch_size']
    C_in = CONFIG['in_channels']
    C_out = CONFIG['out_channels']
    D = CONFIG['depth']
    H = CONFIG['height']
    W = CONFIG['width']
    K = CONFIG['kernel_size']

    x = jax.random.uniform(k0, (B, C_in, D, H, W), dtype=dtype)
    conv_weight = jax.random.normal(k1, (C_in, C_out, K, K, K), dtype=dtype) * 0.02
    conv_bias = jnp.zeros((C_out,), dtype=dtype)
    bias = jax.random.normal(k2, (C_out, 1, 1, 1), dtype=dtype) * 0.02
    return x, conv_weight, conv_bias, bias


def workload(x, conv_weight, conv_bias, bias):
    s = CONFIG['stride']
    p = CONFIG['padding']
    op = CONFIG['output_padding']
    k = conv_weight.shape[2]

    x_ndhwc = jnp.transpose(x, (0, 2, 3, 4, 1))
    w_flipped = jnp.flip(conv_weight, axis=(2, 3, 4))
    w_dhwio = jnp.transpose(w_flipped, (2, 3, 4, 0, 1))

    pad_before = k - 1 - p
    pad_after = k - 1 - p + op
    padding = [(pad_before, pad_after)] * 3

    out_ndhwc = lax.conv_general_dilated(
        x_ndhwc,
        w_dhwio,
        window_strides=(1, 1, 1),
        padding=padding,
        lhs_dilation=(s, s, s),
        dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC'),
    )
    out_ndhwc = out_ndhwc + conv_bias.reshape(1, 1, 1, 1, -1)
    x_conv = jnp.transpose(out_ndhwc, (0, 4, 1, 2, 3))

    original_x = x_conv
    y = x_conv + bias
    y = y + original_x
    y = y * original_x
    y = y + original_x
    return y
