"""[bf16 variant] ComplexConvTranspose2d: two ConvTranspose2d (re/im) combined as complex multiply, stacked on last axis."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'ComplexConvTranspose2d_3769',
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


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG
    k = c['kernel_size']
    keys = jax.random.split(jax.random.key(0), 5)
    x = jax.random.uniform(keys[0], (c['batch'], c['in_channels'], c['height'], c['width'], 2), dtype)
    w_shape = (c['in_channels'], c['out_channels'], k, k)  # PyTorch ConvTranspose2d layout
    w_re = jax.random.normal(keys[1], w_shape, dtype) * 0.02
    b_re = jax.random.normal(keys[2], (c['out_channels'],), dtype) * 0.02
    w_im = jax.random.normal(keys[3], w_shape, dtype) * 0.02
    b_im = jax.random.normal(keys[4], (c['out_channels'],), dtype) * 0.02
    return x, w_re, b_re, w_im, b_im


def _conv_transpose(x_nhwc, w, b):
    c = CONFIG
    s, p, op, d = c['stride'], c['padding'], c['output_padding'], c['dilation']
    kh, kw = w.shape[2], w.shape[3]
    # transposed conv == dilated-input correlation with spatially flipped kernel, (in,out,kh,kw) -> HWIO
    w_hwio = jnp.transpose(w[:, :, ::-1, ::-1], (2, 3, 0, 1))
    pad = [(d * (kh - 1) - p, d * (kh - 1) - p + op), (d * (kw - 1) - p, d * (kw - 1) - p + op)]
    y = lax.conv_general_dilated(
        x_nhwc, w_hwio, window_strides=(1, 1), padding=pad,
        lhs_dilation=(s, s), rhs_dilation=(d, d),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        precision=lax.Precision.HIGHEST,
    )
    return y + b


def workload(x, w_re, b_re, w_im, b_im):
    unbatched = x.ndim == 4  # PyTorch accepts unbatched (C,H,W,2)
    if unbatched:
        x = x[None]
    xr = jnp.transpose(x[..., 0], (0, 2, 3, 1))
    xi = jnp.transpose(x[..., 1], (0, 2, 3, 1))
    real = _conv_transpose(xr, w_re, b_re) - _conv_transpose(xi, w_im, b_im)
    imag = _conv_transpose(xi, w_re, b_re) + _conv_transpose(xr, w_im, b_im)
    out = jnp.stack((real, imag), axis=-1)  # N,H,W,C,2
    out = jnp.transpose(out, (0, 3, 1, 2, 4))  # N,C,H,W,2
    return out[0] if unbatched else out
