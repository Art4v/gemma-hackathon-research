"""Conv2d followed by channel-wise multiplication, LeakyReLU, and GELU."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '54_Conv2d_Multiply_LeakyReLU_GELU',
    'batch_size': 16,
    'in_channels': 128,
    'out_channels': 128,
    'height': 256,
    'width': 256,
    'kernel_size': 3,
    'stride': 1,
    'padding': 'VALID',
    'negative_slope': 0.01,
}

def create_inputs(dtype=jnp.float32):
    k0, k1, k2, k3 = jax.random.split(jax.random.key(0), 4)
    B = CONFIG['batch_size']
    C_in = CONFIG['in_channels']
    C_out = CONFIG['out_channels']
    H = CONFIG['height']
    W = CONFIG['width']
    K = CONFIG['kernel_size']
    
    x = jax.random.uniform(k0, (B, C_in, H, W), dtype=dtype)
    weight = jax.random.normal(k1, (K, K, C_in, C_out), dtype=dtype) * 0.02
    bias = jax.random.normal(k2, (C_out,), dtype=dtype) * 0.02
    multiplier = jax.random.normal(k3, (C_out, 1, 1), dtype=dtype) * 0.02
    return x, weight, bias, multiplier

def workload(x, weight, bias, multiplier):
    s = CONFIG['stride']
    pad = CONFIG['padding']
    slope = CONFIG['negative_slope']
    
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    out_nhwc = lax.conv_general_dilated(
        x_nhwc,
        weight,
        window_strides=(s, s),
        padding=pad,
        dimension_numbers=('NHWC', 'HWIO', 'NHWC')
    )
    out_nhwc = out_nhwc + bias
    out = jnp.transpose(out_nhwc, (0, 3, 1, 2))
    out = out * multiplier
    out = jax.nn.leaky_relu(out, negative_slope=slope)
    out = jax.nn.gelu(out, approximate=False)
    return out
