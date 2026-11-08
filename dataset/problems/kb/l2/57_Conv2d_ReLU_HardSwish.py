"""2D convolution followed by ReLU and HardSwish activations."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '57_Conv2d_ReLU_HardSwish',
    'batch_size': 64,
    'in_channels': 32,
    'out_channels': 128,
    'height': 256,
    'width': 256,
    'kernel_size': 3,
    'stride': 1,
    'hardswish_offset': 3.0,
    'hardswish_scale': 6.0,
    'clamp_min': 0.0,
    'clamp_max': 1.0,
}

def create_inputs(dtype=jnp.float32):
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
    return (x, weight, bias)

def workload(x, weight, bias):
    s = CONFIG['stride']
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    w_hwio = jnp.transpose(weight, (2, 3, 1, 0))
    
    out_nhwc = lax.conv_general_dilated(
        x_nhwc,
        w_hwio,
        window_strides=(s, s),
        padding='VALID',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
    )
    out_nhwc = out_nhwc + bias.reshape(1, 1, 1, -1)
    out = jnp.transpose(out_nhwc, (0, 3, 1, 2))
    
    out = jax.nn.relu(out)
    out = out * jnp.clip(
        (out + CONFIG['hardswish_offset']) / CONFIG['hardswish_scale'],
        CONFIG['clamp_min'],
        CONFIG['clamp_max'],
    )
    return out
