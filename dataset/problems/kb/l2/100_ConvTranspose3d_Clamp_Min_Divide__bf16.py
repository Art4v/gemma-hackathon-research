"""[bf16 variant] 3D transposed convolution followed by minimum clamp and division."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '100_ConvTranspose3d_Clamp_Min_Divide',
    'batch_size': 16,
    'in_channels': 128,
    'out_channels': 128,
    'depth': 16,
    'height': 32,
    'width': 32,
    'kernel_size': 3,
    'stride': 2,
    'padding': 1,
    'min_value': -1.0,
    'divisor': 2.0,
}

def create_inputs(dtype=jnp.bfloat16):
    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
    B = CONFIG['batch_size']
    C_in = CONFIG['in_channels']
    C_out = CONFIG['out_channels']
    D = CONFIG['depth']
    H = CONFIG['height']
    W = CONFIG['width']
    K = CONFIG['kernel_size']
    x = jax.random.uniform(k1, (B, C_in, D, H, W), dtype=dtype)
    weight = jax.random.normal(k2, (C_in, C_out, K, K, K), dtype=dtype) * 0.02
    bias = jax.random.normal(k3, (C_out,), dtype=dtype) * 0.02
    return (x, weight, bias)

def workload(x, weight, bias):
    s = CONFIG['stride']
    p = CONFIG['padding']
    k = weight.shape[2]
    pad = k - 1 - p
    x_ndhwc = jnp.transpose(x, (0, 2, 3, 4, 1))
    w_flipped = jnp.flip(weight, axis=(2, 3, 4))
    w_dhwio = jnp.transpose(w_flipped, (2, 3, 4, 0, 1))
    out = lax.conv_general_dilated(
        lhs=x_ndhwc,
        rhs=w_dhwio,
        window_strides=(1, 1, 1),
        padding=[(pad, pad), (pad, pad), (pad, pad)],
        lhs_dilation=(s, s, s),
        dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC'),
    )
    out = out + bias
    out = jnp.transpose(out, (0, 4, 1, 2, 3))
    out = jnp.maximum(out, CONFIG['min_value'])
    out = out / CONFIG['divisor']
    return out
