"""[bf16 variant] 3D convolution followed by ReLU, LeakyReLU, GELU, Sigmoid activations, and bias addition."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '7_Conv3d_ReLU_LeakyReLU_GELU_Sigmoid_BiasAdd',
    'batch_size': 64,
    'in_channels': 8,
    'out_channels': 32,
    'depth': 32,
    'height': 64,
    'width': 64,
    'kernel_size': 3,
    'negative_slope': 0.01,
}

def create_inputs(dtype=jnp.bfloat16):
    k0, k1, k2, k3 = jax.random.split(jax.random.key(0), 4)
    B = CONFIG['batch_size']
    C_in = CONFIG['in_channels']
    C_out = CONFIG['out_channels']
    D = CONFIG['depth']
    H = CONFIG['height']
    W = CONFIG['width']
    K = CONFIG['kernel_size']

    x = jax.random.uniform(k0, (B, C_in, D, H, W), dtype=dtype)
    conv_weight = jax.random.normal(k1, (C_out, C_in, K, K, K), dtype=dtype) * 0.02
    conv_bias = jax.random.normal(k2, (C_out,), dtype=dtype) * 0.02
    bias = jax.random.normal(k3, (C_out, 1, 1, 1), dtype=dtype) * 0.02
    return x, conv_weight, conv_bias, bias

def workload(x, conv_weight, conv_bias, bias):
    x_ndhwc = jnp.transpose(x, (0, 2, 3, 4, 1))
    w_dhwio = jnp.transpose(conv_weight, (2, 3, 4, 1, 0))
    out = lax.conv_general_dilated(
        x_ndhwc,
        w_dhwio,
        window_strides=(1, 1, 1),
        padding='VALID',
        dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC'),
    )
    out = out + conv_bias
    out = jax.nn.relu(out)
    out = jax.nn.leaky_relu(out, negative_slope=CONFIG['negative_slope'])
    out = jax.nn.gelu(out, approximate=False)
    out = jax.nn.sigmoid(out)
    out = jnp.transpose(out, (0, 4, 1, 2, 3))
    out = out + bias
    return out
