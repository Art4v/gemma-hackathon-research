"""[bf16 variant] Conv2d followed by constant division and LeakyReLU activation."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '71_Conv2d_Divide_LeakyReLU',
    'batch_size': 128,
    'in_channels': 64,
    'out_channels': 128,
    'height': 128,
    'width': 128,
    'kernel_size': 3,
    'divisor': 2.0,
    'negative_slope': 0.01,
}

def create_inputs(dtype=jnp.bfloat16):
    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
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
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    w_hwio = jnp.transpose(weight, (2, 3, 1, 0))
    out_nhwc = lax.conv_general_dilated(
        x_nhwc,
        w_hwio,
        window_strides=(1, 1),
        padding='VALID',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC')
    )
    out_nhwc = out_nhwc + jnp.reshape(bias, (1, 1, 1, -1))
    out = jnp.transpose(out_nhwc, (0, 3, 1, 2))
    out = out / CONFIG['divisor']
    out = jax.nn.leaky_relu(out, negative_slope=CONFIG['negative_slope'])
    return out
