"""Gated transposed conv: ConvTranspose2d(x) * sigmoid(ConvTranspose2d(x))."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'GatedConvTranspose_2141',
    'batch': 8,
    'in_channels': 128,
    'out_channels': 128,
    'height': 128,
    'width': 128,
    'kernel_size': 4,
    'stride': 1,
    'padding': 0,
    'output_padding': 0,
    'dilation': 1,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    k = c['kernel_size']
    kx, kwf, kbf, kwg, kbg = jax.random.split(jax.random.key(0), 5)
    x = jax.random.uniform(kx, (c['batch'], c['in_channels'], c['height'], c['width']), dtype)
    w_shape = (c['in_channels'], c['out_channels'], k, k)  # PyTorch ConvTranspose2d layout
    w_f = jax.random.normal(kwf, w_shape, dtype) * 0.02
    b_f = jax.random.normal(kbf, (c['out_channels'],), dtype) * 0.02
    w_g = jax.random.normal(kwg, w_shape, dtype) * 0.02
    b_g = jax.random.normal(kbg, (c['out_channels'],), dtype) * 0.02
    return x, w_f, b_f, w_g, b_g


def _conv_transpose(x_nhwc, w, b):
    # Transposed conv as a regular conv: flip the kernel spatially, (Cin, Cout, kH, kW) -> HWIO,
    # pad by d*(k-1)-p (plus output_padding on the high side), dilate the input by stride.
    c = CONFIG
    k, s, p, op, d = c['kernel_size'], c['stride'], c['padding'], c['output_padding'], c['dilation']
    w_hwio = jnp.transpose(jnp.flip(w, axis=(2, 3)), (2, 3, 0, 1))
    lo = d * (k - 1) - p
    hi = lo + op
    y = lax.conv_general_dilated(
        x_nhwc, w_hwio,
        window_strides=(1, 1),
        padding=((lo, hi), (lo, hi)),
        lhs_dilation=(s, s),
        rhs_dilation=(d, d),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
    )
    return y + b


def workload(x, w_f, b_f, w_g, b_g):
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    f = _conv_transpose(x_nhwc, w_f, b_f)
    g = jax.nn.sigmoid(_conv_transpose(x_nhwc, w_g, b_g))
    return jnp.transpose(f * g, (0, 3, 1, 2))
