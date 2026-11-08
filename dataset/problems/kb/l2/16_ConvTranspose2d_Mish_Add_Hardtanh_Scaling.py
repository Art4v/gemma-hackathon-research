"""ConvTranspose2d, Mish activation, scalar addition, Hardtanh activation, and scalar scaling."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '16_ConvTranspose2d_Mish_Add_Hardtanh_Scaling',
    'batch_size': 128,
    'in_channels': 64,
    'out_channels': 64,
    'height': 128,
    'width': 128,
    'kernel_size': 3,
    'stride': 2,
    'padding': 1,
    'output_padding': 1,
    'add_value': 0.5,
    'scale': 2.0,
    'hardtanh_min': -1.0,
    'hardtanh_max': 1.0,
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
    weight = jax.random.normal(k2, (C_in, C_out, K, K), dtype=dtype) * 0.02
    bias = jax.random.normal(k3, (C_out,), dtype=dtype) * 0.02
    return (x, weight, bias)

def workload(x, weight, bias):
    stride = CONFIG['stride']
    padding = CONFIG['padding']
    output_padding = CONFIG['output_padding']
    add_value = CONFIG['add_value']
    scale = CONFIG['scale']
    min_val = CONFIG['hardtanh_min']
    max_val = CONFIG['hardtanh_max']
    
    kh, kw = weight.shape[2], weight.shape[3]
    pad_h = (kh - 1 - padding, kh - 1 - padding + output_padding)
    pad_w = (kw - 1 - padding, kw - 1 - padding + output_padding)
    
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    w_hwoi = jnp.flip(jnp.transpose(weight, (2, 3, 1, 0)), axis=(0, 1))
    
    out_nhwc = lax.conv_general_dilated(
        lhs=x_nhwc,
        rhs=w_hwoi,
        window_strides=(1, 1),
        padding=(pad_h, pad_w),
        lhs_dilation=(stride, stride),
        dimension_numbers=('NHWC', 'HWOI', 'NHWC'),
    )
    out_nhwc = out_nhwc + bias.reshape(1, 1, 1, -1)
    out = jnp.transpose(out_nhwc, (0, 3, 1, 2))
    
    out = out * jnp.tanh(jax.nn.softplus(out))
    out = out + add_value
    out = jnp.clip(out, min_val, max_val)
    out = out * scale
    return out
