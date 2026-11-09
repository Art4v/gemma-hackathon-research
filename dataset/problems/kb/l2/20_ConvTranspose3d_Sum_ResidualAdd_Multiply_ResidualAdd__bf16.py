"""[bf16 variant] ConvTranspose3d -> add bias -> residual add -> multiply -> residual add."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '20_ConvTranspose3d_Sum_ResidualAdd_Multiply_ResidualAdd',
    'batch_size': 32,
    'in_channels': 32,
    'out_channels': 64,
    'depth': 16,
    'height': 32,
    'width': 32,
    'kernel_size': 3,
    'stride': 2,
    'padding': 1,
    'output_padding': 1,
    'bias_shape': (64, 1, 1, 1),
}


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG
    key = jax.random.key(0)
    kx, kw, kb, kbias = jax.random.split(key, 4)
    k = c['kernel_size']
    x = jax.random.uniform(
        kx, (c['batch_size'], c['in_channels'], c['depth'], c['height'], c['width']), dtype)
    # PyTorch ConvTranspose3d weight layout: (in, out, kD, kH, kW)
    w = jax.random.normal(kw, (c['in_channels'], c['out_channels'], k, k, k), dtype) * 0.02
    conv_b = jax.random.normal(kb, (c['out_channels'],), dtype) * 0.02
    bias = jax.random.normal(kbias, c['bias_shape'], dtype) * 0.02
    return x, w, conv_b, bias


def workload(x, w, conv_b, bias):
    s = CONFIG['stride']
    p = CONFIG['padding']
    op = CONFIG['output_padding']
    k = w.shape[2]
    x_ndhwc = jnp.transpose(x, (0, 2, 3, 4, 1))
    # (in, out, kD, kH, kW) -> (kD, kH, kW, in, out), spatially flipped
    kern = jnp.transpose(w, (2, 3, 4, 0, 1))[::-1, ::-1, ::-1, :, :]
    lo = k - 1 - p
    hi = k - 1 - p + op
    y = lax.conv_general_dilated(
        x_ndhwc, kern,
        window_strides=(1, 1, 1),
        padding=((lo, hi), (lo, hi), (lo, hi)),
        lhs_dilation=(s, s, s),
        rhs_dilation=(1, 1, 1),
        dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC'),
        precision=lax.Precision.HIGHEST,
    )
    y = y + conv_b
    orig = jnp.transpose(y, (0, 4, 1, 2, 3))
    out = orig + bias
    out = out + orig
    out = out * orig
    out = out + orig
    return out
