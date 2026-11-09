"""[bf16 variant] 3D convolution followed by scaling, tanh, multiplication, and sigmoid."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '48_Conv3d_Scaling_Tanh_Multiply_Sigmoid',
    'batch_size': 16,
    'in_channels': 32,
    'out_channels': 64,
    'depth': 32,
    'height': 64,
    'width': 64,
    'kernel_size': 3,
}

def create_inputs(dtype=jnp.bfloat16):
    k0, k1, k2, k3, k4 = jax.random.split(jax.random.key(0), 5)
    B = CONFIG['batch_size']
    C_in = CONFIG['in_channels']
    C_out = CONFIG['out_channels']
    D = CONFIG['depth']
    H = CONFIG['height']
    W = CONFIG['width']
    K = CONFIG['kernel_size']

    x = jax.random.uniform(k0, (B, C_in, D, H, W), dtype=dtype)
    weight = jax.random.normal(k1, (C_out, C_in, K, K, K), dtype=dtype) * 0.02
    conv_bias = jax.random.normal(k2, (C_out,), dtype=dtype) * 0.02
    scaling_factor = jax.random.normal(k3, (C_out, 1, 1, 1), dtype=dtype)
    bias = jax.random.normal(k4, (C_out, 1, 1, 1), dtype=dtype)
    return x, weight, conv_bias, scaling_factor, bias

def workload(x, weight, conv_bias, scaling_factor, bias):
    x_ndhwc = jnp.transpose(x, (0, 2, 3, 4, 1))
    w_dhwio = jnp.transpose(weight, (2, 3, 4, 1, 0))
    out = lax.conv_general_dilated(
        lhs=x_ndhwc,
        rhs=w_dhwio,
        window_strides=(1, 1, 1),
        padding='VALID',
        dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC'),
    )
    out = out + conv_bias
    x = jnp.transpose(out, (0, 4, 1, 2, 3))
    x = x * scaling_factor
    x = jnp.tanh(x)
    x = x * bias
    x = jax.nn.sigmoid(x)
    return x
