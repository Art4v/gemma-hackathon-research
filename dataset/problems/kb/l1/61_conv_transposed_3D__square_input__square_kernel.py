"""ConvTranspose3d (square input, square kernel, stride 1, no bias)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '61_conv_transposed_3D__square_input__square_kernel',
    'batch_size': 8,
    'in_channels': 48,
    'out_channels': 48,
    'kernel_size': 3,
    'depth': 64,
    'height': 64,
    'width': 64,
    'stride': 1,
    'padding': 0,
    'output_padding': 0,
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
    op = CONFIG['output_padding']
    k = w.shape[2]
    # (in, out, kd, kh, kw) -> flipped, DHWIO
    w_t = jnp.flip(w, axis=(2, 3, 4))
    w_t = jnp.transpose(w_t, (2, 3, 4, 0, 1))
    x_n = jnp.transpose(x, (0, 2, 3, 4, 1))  # NCDHW -> NDHWC
    pad = [(k - 1 - p, k - 1 - p + op)] * 3
    y = lax.conv_general_dilated(
        x_n, w_t,
        window_strides=(1, 1, 1),
        padding=pad,
        lhs_dilation=(s, s, s),
        rhs_dilation=(1, 1, 1),
        dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC'),
    )
    return jnp.transpose(y, (0, 4, 1, 2, 3))  # NDHWC -> NCDHW
