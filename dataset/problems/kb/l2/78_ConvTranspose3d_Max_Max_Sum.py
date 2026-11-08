"""ConvTranspose3d -> MaxPool3d(2) -> MaxPool3d(3) -> sum over channels (keepdim)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '78_ConvTranspose3d_Max_Max_Sum',
    'batch_size': 16,
    'in_channels': 32,
    'out_channels': 64,
    'depth': 32,
    'height': 32,
    'width': 32,
    'kernel_size': 5,
    'stride': 2,
    'padding': 2,
    'output_padding': 0,
    'pool1_kernel': 2,
    'pool2_kernel': 3,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    kx, kw, kb = jax.random.split(key, 3)
    c = CONFIG
    x = jax.random.uniform(
        kx, (c['batch_size'], c['in_channels'], c['depth'], c['height'], c['width']), dtype)
    k = c['kernel_size']
    # PyTorch ConvTranspose3d weight layout: (Cin, Cout, kD, kH, kW)
    w = jax.random.normal(kw, (c['in_channels'], c['out_channels'], k, k, k), dtype) * 0.02
    b = jax.random.normal(kb, (c['out_channels'],), dtype) * 0.02
    return x, w, b


def _max_pool3d_ndhwc(x, k):
    return lax.reduce_window(
        x, -jnp.inf, lax.max,
        window_dimensions=(1, k, k, k, 1),
        window_strides=(1, k, k, k, 1),
        padding='VALID')


def workload(x, w, b):
    c = CONFIG
    k = w.shape[2]
    s = c['stride']
    p = c['padding']
    op = c['output_padding']
    # NCDHW -> NDHWC
    xh = jnp.transpose(x, (0, 2, 3, 4, 1))
    # (Cin, Cout, kD, kH, kW) -> flipped DHWIO
    wk = jnp.transpose(w[:, :, ::-1, ::-1, ::-1], (2, 3, 4, 0, 1))
    pad = [(k - 1 - p, k - 1 - p + op)] * 3
    y = lax.conv_general_dilated(
        xh, wk,
        window_strides=(1, 1, 1),
        padding=pad,
        lhs_dilation=(s, s, s),
        rhs_dilation=(1, 1, 1),
        dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC'))
    y = y + b
    y = _max_pool3d_ndhwc(y, c['pool1_kernel'])
    y = _max_pool3d_ndhwc(y, c['pool2_kernel'])
    y = jnp.sum(y, axis=-1, keepdims=True)
    # NDHWC -> NCDHW
    return jnp.transpose(y, (0, 4, 1, 2, 3))
