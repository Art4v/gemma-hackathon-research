"""[bf16 variant] 3D transposed convolution with padding, dilation, and stride."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '77_conv_transposed_3D_square_input_square_kernel___padded____dilated____strided__',
    'batch_size': 16,
    'in_channels': 128,
    'out_channels': 128,
    'kernel_size': 3,
    'depth': 16,
    'height': 32,
    'width': 32,
    'stride': 2,
    'padding': 1,
    'dilation': 2,
}

def create_inputs(dtype=jnp.bfloat16):
    k1, k2 = jax.random.split(jax.random.key(0), 2)
    B = CONFIG['batch_size']
    C_in = CONFIG['in_channels']
    C_out = CONFIG['out_channels']
    K = CONFIG['kernel_size']
    D = CONFIG['depth']
    H = CONFIG['height']
    W = CONFIG['width']
    x = jax.random.uniform(k1, (B, C_in, D, H, W), dtype=dtype)
    weight = jax.random.normal(k2, (C_in, C_out, K, K, K), dtype=dtype) * 0.02
    return (x, weight)

def workload(x, weight):
    stride = CONFIG['stride']
    padding = CONFIG['padding']
    dilation = CONFIG['dilation']
    K = weight.shape[2]

    x_ndhwc = jnp.transpose(x, (0, 2, 3, 4, 1))
    w_flipped = jnp.flip(weight, axis=(2, 3, 4))
    w_dhwoi = jnp.transpose(w_flipped, (2, 3, 4, 1, 0))

    k_eff = (K - 1) * dilation + 1
    pad = k_eff - 1 - padding
    jax_padding = ((pad, pad), (pad, pad), (pad, pad))

    out_ndhwc = lax.conv_general_dilated(
        lhs=x_ndhwc,
        rhs=w_dhwoi,
        window_strides=(1, 1, 1),
        padding=jax_padding,
        lhs_dilation=(stride, stride, stride),
        rhs_dilation=(dilation, dilation, dilation),
        dimension_numbers=('NDHWC', 'DHWOI', 'NDHWC'),
    )
    return jnp.transpose(out_ndhwc, (0, 4, 1, 2, 3))
