"""Complex ConvTranspose2d: (Tr(xr) - Ti(xi), Tr(xi) + Ti(xr))."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'ComplexConvTranspose2d_12328',
    'batch': 16,
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
    keys = jax.random.split(jax.random.key(0), 6)
    x_shape = (c['batch'], c['in_channels'], c['height'], c['width'])
    w_shape = (c['in_channels'], c['out_channels'], c['kernel_size'], c['kernel_size'])
    x_r = jax.random.uniform(keys[0], x_shape, dtype)
    x_i = jax.random.uniform(keys[1], x_shape, dtype)
    w_r = jax.random.normal(keys[2], w_shape, dtype) * 0.02
    b_r = jax.random.normal(keys[3], (c['out_channels'],), dtype) * 0.02
    w_i = jax.random.normal(keys[4], w_shape, dtype) * 0.02
    b_i = jax.random.normal(keys[5], (c['out_channels'],), dtype) * 0.02
    return x_r, x_i, w_r, b_r, w_i, b_i


def _conv_transpose(x, w, b):
    # x: NCHW, w: PyTorch ConvTranspose2d layout (in, out, kH, kW), groups=1.
    s, p = CONFIG['stride'], CONFIG['padding']
    op, d = CONFIG['output_padding'], CONFIG['dilation']
    kh, kw = w.shape[2], w.shape[3]
    pad_h = d * (kh - 1) - p
    pad_w = d * (kw - 1) - p
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    kern = jnp.transpose(w[:, :, ::-1, ::-1], (2, 3, 0, 1))  # HWIO, flipped
    y = lax.conv_general_dilated(
        x_nhwc, kern,
        window_strides=(1, 1),
        padding=[(pad_h, pad_h + op), (pad_w, pad_w + op)],
        lhs_dilation=(s, s),
        rhs_dilation=(d, d),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        precision=lax.Precision.HIGHEST,
    )
    y = y + b
    return jnp.transpose(y, (0, 3, 1, 2))


def workload(x_r, x_i, w_r, b_r, w_i, b_i):
    out_r = _conv_transpose(x_r, w_r, b_r) - _conv_transpose(x_i, w_i, b_i)
    out_i = _conv_transpose(x_i, w_r, b_r) + _conv_transpose(x_r, w_i, b_i)
    return out_r, out_i
