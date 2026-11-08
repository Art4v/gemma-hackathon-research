"""ConvTranspose3d -> clamp(min) -> divide by constant."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '100_ConvTranspose3d_Clamp_Min_Divide',
    'batch_size': 16,
    'in_channels': 64,
    'out_channels': 128,
    'depth': 24,
    'height': 48,
    'width': 48,
    'kernel_size': 3,
    'stride': 2,
    'padding': 1,
    'output_padding': 0,
    'min_value': -1.0,
    'divisor': 2.0,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    kx, kw, kb = jax.random.split(key, 3)
    c = CONFIG
    k = c['kernel_size']
    x = jax.random.uniform(
        kx, (c['batch_size'], c['in_channels'], c['depth'], c['height'], c['width']), dtype)
    # PyTorch ConvTranspose3d weight layout: (in, out, kD, kH, kW)
    w = jax.random.normal(kw, (c['in_channels'], c['out_channels'], k, k, k), dtype) * 0.02
    b = jax.random.normal(kb, (c['out_channels'],), dtype) * 0.02
    return x, w, b


def workload(x, w, b):
    c = CONFIG
    k = w.shape[2]
    s = c['stride']
    p = c['padding']
    op = c['output_padding']
    # NCDHW -> NDHWC
    x_nd = jnp.transpose(x, (0, 2, 3, 4, 1))
    # (in, out, kD, kH, kW) -> flipped spatially -> (kD, kH, kW, in, out)
    w_f = jnp.flip(w, axis=(2, 3, 4))
    w_dhwio = jnp.transpose(w_f, (2, 3, 4, 0, 1))
    pad = [(k - 1 - p, k - 1 - p + op)] * 3
    y = lax.conv_general_dilated(
        x_nd, w_dhwio,
        window_strides=(1, 1, 1),
        padding=pad,
        lhs_dilation=(s, s, s),
        rhs_dilation=(1, 1, 1),
        dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC'),
    )
    y = y + b.reshape(1, 1, 1, 1, -1)
    y = jnp.maximum(y, jnp.asarray(c['min_value'], y.dtype))
    y = y / jnp.asarray(c['divisor'], y.dtype)
    # NDHWC -> NCDHW
    return jnp.transpose(y, (0, 4, 1, 2, 3))
