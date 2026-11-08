"""ConvTranspose3d (square kernel, stride, padding, dilation, no bias)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '77_conv_transposed_3D_square_input_square_kernel___padded____dilated____strided__',
    'batch_size': 8,
    'in_channels': 128,
    'out_channels': 128,
    'kernel_size': 3,
    'depth': 16,
    'height': 32,
    'width': 32,
    'stride': 2,
    'padding': 1,
    'dilation': 2,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    kx, kw = jax.random.split(key)
    c = CONFIG
    x = jax.random.uniform(
        kx, (c['batch_size'], c['in_channels'], c['depth'], c['height'], c['width']), dtype)
    k = c['kernel_size']
    w = jax.random.normal(kw, (c['in_channels'], c['out_channels'], k, k, k), dtype) * 0.02
    return x, w


def workload(x, w):
    s = CONFIG['stride']
    p = CONFIG['padding']
    d = CONFIG['dilation']
    k = w.shape[2]
    pad = d * (k - 1) - p
    x_ndhwc = jnp.transpose(x, (0, 2, 3, 4, 1))
    # (Cin, Cout, kD, kH, kW) -> flip spatial -> (kD, kH, kW, Cin, Cout)
    rhs = jnp.transpose(jnp.flip(w, axis=(2, 3, 4)), (2, 3, 4, 0, 1))
    y = lax.conv_general_dilated(
        x_ndhwc, rhs,
        window_strides=(1, 1, 1),
        padding=[(pad, pad)] * 3,
        lhs_dilation=(s, s, s),
        rhs_dilation=(d, d, d),
        dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC'),
        precision=lax.Precision.HIGHEST,
    )
    return jnp.transpose(y, (0, 4, 1, 2, 3))
