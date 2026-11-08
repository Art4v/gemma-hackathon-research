"""AvgPool3d -> ConvTranspose3d -> clamp -> spatial softmax -> per-channel scale multiply."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '38_ConvTranspose3d_AvgPool_Clamp_Softmax_Multiply',
    'batch_size': 32,
    'in_channels': 32,
    'out_channels': 64,
    'depth': 32,
    'height': 64,
    'width': 64,
    'kernel_size': 3,
    'stride': 2,
    'padding': 1,
    'output_padding': 1,
    'pool_kernel_size': 2,
    'clamp_min': 0.0,
    'clamp_max': 1.0,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    key = jax.random.key(0)
    kx, kw, kb = jax.random.split(key, 3)
    x = jax.random.uniform(
        kx, (c['batch_size'], c['in_channels'], c['depth'], c['height'], c['width']), dtype)
    k = c['kernel_size']
    # PyTorch ConvTranspose3d weight layout: (in, out, kD, kH, kW)
    w = jax.random.normal(kw, (c['in_channels'], c['out_channels'], k, k, k), dtype) * 0.02
    b = jax.random.normal(kb, (c['out_channels'],), dtype) * 0.02
    scale = jnp.ones((1, c['out_channels'], 1, 1, 1), dtype)
    return x, w, b, scale


def workload(x, w, b, scale):
    c = CONFIG
    p = c['pool_kernel_size']
    k = c['kernel_size']
    s = c['stride']
    pad = c['padding']
    op = c['output_padding']

    # NCDHW -> NDHWC
    x = jnp.transpose(x, (0, 2, 3, 4, 1))
    n, d, h, wd, ci = x.shape
    dp, hp, wp = d // p, h // p, wd // p
    x = x[:, :dp * p, :hp * p, :wp * p, :]
    x = x.reshape(n, dp, p, hp, p, wp, p, ci).mean(axis=(2, 4, 6))

    # Transposed conv as dilated-input conv with flipped kernel.
    kern = jnp.flip(w, axis=(2, 3, 4))
    kern = jnp.transpose(kern, (2, 3, 4, 0, 1))  # DHWIO
    lo = k - 1 - pad
    hi = k - 1 - pad + op
    y = lax.conv_general_dilated(
        x, kern,
        window_strides=(1, 1, 1),
        padding=((lo, hi), (lo, hi), (lo, hi)),
        lhs_dilation=(s, s, s),
        rhs_dilation=(1, 1, 1),
        dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC'),
    )
    y = y + b.reshape(1, 1, 1, 1, -1)

    # NDHWC -> NCDHW
    y = jnp.transpose(y, (0, 4, 1, 2, 3))
    y = jnp.clip(y, c['clamp_min'], c['clamp_max'])
    bb, cc, dd, hh, ww = y.shape
    y = jax.nn.softmax(y.reshape(bb, cc, dd * hh * ww), axis=2)
    y = y.reshape(bb, cc, dd, hh, ww)
    return y * scale
