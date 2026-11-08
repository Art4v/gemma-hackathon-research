"""ConvTranspose2d, bias add, clamp, scaling, clamp, and divide."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '2_ConvTranspose2d_BiasAdd_Clamp_Scaling_Clamp_Divide',
    'batch_size': 64,
    'in_channels': 128,
    'out_channels': 128,
    'height': 128,
    'width': 128,
    'kernel_size': 3,
    'stride': 2,
    'padding': 1,
    'output_padding': 1,
    'scaling_factor': 2.0,
    'clamp_min': 0.0,
    'clamp_max': 1.0,
}

def create_inputs(dtype=jnp.float32):
    k1, k2, k3, k4 = jax.random.split(jax.random.key(0), 4)
    B = CONFIG['batch_size']
    C_in = CONFIG['in_channels']
    C_out = CONFIG['out_channels']
    H = CONFIG['height']
    W = CONFIG['width']
    K = CONFIG['kernel_size']

    x = jax.random.uniform(k1, (B, C_in, H, W), dtype=dtype)
    weight = jax.random.normal(k2, (C_in, C_out, K, K), dtype=dtype) * 0.02
    conv_bias = jax.random.normal(k3, (C_out,), dtype=dtype) * 0.02
    bias = jax.random.normal(k4, (C_out, 1, 1), dtype=dtype) * 0.02
    return (x, weight, conv_bias, bias)

def workload(x, weight, conv_bias, bias):
    stride = CONFIG['stride']
    padding = CONFIG['padding']
    output_padding = CONFIG['output_padding']
    scaling_factor = CONFIG['scaling_factor']
    clamp_min = CONFIG['clamp_min']
    clamp_max = CONFIG['clamp_max']

    kh, kw = weight.shape[2], weight.shape[3]
    pad_h_before = kh - 1 - padding
    pad_h_after = kh - 1 - padding + output_padding
    pad_w_before = kw - 1 - padding
    pad_w_after = kw - 1 - padding + output_padding

    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    w_hwio = jnp.transpose(weight, (2, 3, 0, 1))[::-1, ::-1, :, :]

    out_nhwc = lax.conv_general_dilated(
        lhs=x_nhwc,
        rhs=w_hwio,
        window_strides=(1, 1),
        padding=((pad_h_before, pad_h_after), (pad_w_before, pad_w_after)),
        lhs_dilation=(stride, stride),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
    )
    out_nhwc = out_nhwc + conv_bias
    out = jnp.transpose(out_nhwc, (0, 3, 1, 2))

    out = out + bias
    out = jnp.clip(out, clamp_min, clamp_max)
    out = out * scaling_factor
    out = jnp.clip(out, clamp_min, clamp_max)
    out = out / scaling_factor
    return out
