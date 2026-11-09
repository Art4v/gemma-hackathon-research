"""[bf16 variant] 3D convolution followed by multiplication, InstanceNorm3d, clamping, multiplication, and channel-wise max."""

import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '79_Conv3d_Multiply_InstanceNorm_Clamp_Multiply_Max',
    'batch_size': 64,
    'in_channels': 16,
    'out_channels': 32,
    'depth': 32,
    'height': 64,
    'width': 64,
    'kernel_size': 3,
    'clamp_min': -1.0,
    'clamp_max': 1.0,
    'eps': 1e-5,
}

def create_inputs(dtype=jnp.bfloat16):
    k1, k2, k3, k4 = jax.random.split(jax.random.key(0), 4)
    B = CONFIG['batch_size']
    C_in = CONFIG['in_channels']
    C_out = CONFIG['out_channels']
    D = CONFIG['depth']
    H = CONFIG['height']
    W = CONFIG['width']
    K = CONFIG['kernel_size']

    x = jax.random.uniform(k1, (B, C_in, D, H, W), dtype=dtype)
    weight = jax.random.normal(k2, (C_out, C_in, K, K, K), dtype=dtype) * 0.02
    bias = jax.random.normal(k3, (C_out,), dtype=dtype) * 0.02
    multiplier = jax.random.normal(k4, (C_out, 1, 1, 1), dtype=dtype) * 0.02
    return x, weight, bias, multiplier

def workload(x, weight, bias, multiplier):
    # x: (N, C_in, D, H, W) -> (N, D, H, W, C_in)
    x_ndhwc = jnp.transpose(x, (0, 2, 3, 4, 1))
    # weight: (C_out, C_in, Kd, Kh, Kw) -> (Kd, Kh, Kw, C_in, C_out)
    w_dhwio = jnp.transpose(weight, (2, 3, 4, 1, 0))

    out_ndhwc = lax.conv_general_dilated(
        x_ndhwc,
        w_dhwio,
        window_strides=(1, 1, 1),
        padding='VALID',
        dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC'),
    )
    out_ndhwc = out_ndhwc + bias
    x = jnp.transpose(out_ndhwc, (0, 4, 1, 2, 3))

    # Multiply
    x = x * multiplier

    # InstanceNorm3d (affine=False, eps=1e-5)
    mean = jnp.mean(x, axis=(2, 3, 4), keepdims=True)
    var = jnp.var(x, axis=(2, 3, 4), keepdims=True)
    x = (x - mean) / jnp.sqrt(var + CONFIG['eps'])

    # Clamp
    x = jnp.clip(x, CONFIG['clamp_min'], CONFIG['clamp_max'])

    # Multiply
    x = x * multiplier

    # Max over channel dimension (dim=1)
    x = jnp.max(x, axis=1)
    return x
